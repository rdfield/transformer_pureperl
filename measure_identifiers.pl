#!/usr/bin/perl
# Measure IDENTIFIER FIDELITY: do PO numbers, quote refs, SKUs and serials survive
# summarisation intact, or do they start right and drift?
#
# WHY THIS EXISTS. The rare-token copy-recall metric cannot see the failure that
# actually matters here. It asks "does this token appear ANYWHERE in the source?",
# which is a set-membership test, while the observed failure is about ORDER:
#
#     source   PO1234567890123
#     summary  PO123456789654321      <- right for 9 digits, then drifts
#
# BPE splits that into several tokens, and the drifted pieces are digit groups that
# almost certainly occur somewhere in a long thread. So a completely wrong PO number
# scores near 100% on copy recall. That is why copy recall stopped discriminating:
# not because the problem was solved, but because it never measured it.
#
# Why identifiers specifically fail: a 13-digit number is 4-6 tokens, each predicted
# separately, and digits carry no linguistic constraint — after "123456789" nothing
# makes "0" likelier than "6". Even at 95% per token, five tokens is ~77% end to
# end, and partial credit is worthless: PO123456789654321 is as useless as a blank.
# That also explains why sticky copy helped without fixing it — it raises the
# PROBABILITY of continuing from the adjacent source position, and an identifier
# needs certainty.
#
# Every summary identifier is sorted into one of three buckets against the source:
#   EXACT    — identical to a source identifier
#   DRIFT    — shares >= --min-prefix opening characters with one, but differs
#   INVENTED — no source identifier shares even that much
# For DRIFT the script reports how many characters were right before it went wrong,
# which is the "started off well then tailed off" effect turned into a number.
#
# Inputs use the bake-off formats, so the teacher baseline is available immediately:
#   pairs.csv  id,band,body_b64,summary_b64   (bakeoff_sample.pl output)
#   gen*.csv   id,model,summary_b64,...       (bakeoff_generate.pl output)
# The pairs file supplies the SOURCE for every arm, plus the teacher as arm 1.
#
#   perl measure_identifiers.pl pairs.csv [gen1.csv ...]
#        [--min-prefix 4] [--min-len 5] [--show 12] [--dates]
#
use Modern::Perl;
use Getopt::Long;
use lib '.';
use Text::CSV;
use MIME::Base64 qw(decode_base64);
# Shared with repair_identifiers.pl ON PURPOSE: if the two disagreed about what an
# identifier is, measuring the repair's effect would count different populations.
use ML::Identifiers qw(identifiers best_prefix_match common_prefix);

my $MIN_PREFIX = 4;    # chars of agreement before we call it drift rather than invention
my $MIN_LEN    = 5;    # shortest string we treat as an identifier
my $SHOW       = 12;   # worst drifts to print
my $DATES      = 0;    # include ISO-ish dates (a different failure; off by default)
GetOptions('min-prefix=i' => \$MIN_PREFIX, 'min-len=i' => \$MIN_LEN,
           'show=i' => \$SHOW, 'dates' => \$DATES) or die usage();

my $pairs = shift @ARGV or die usage();
my @gens  = @ARGV;
die "$pairs not found\n" unless -e $pairs;

my $csv = Text::CSV->new({ binary => 1 });
open my $pfh, '<', $pairs or die "$pairs: $!";
my (@ids, %src, %teacher);
while (my $r = $csv->getline($pfh)) {
   next unless @$r >= 4;
   push @ids, $r->[0];
   $src{ $r->[0] }     = b64($r->[2]);
   $teacher{ $r->[0] } = b64($r->[3]);
}
close $pfh;
die "no pairs read\n" unless @ids;
say "source threads: " . scalar(@ids);

# identifiers in each source, extracted once
my %src_ids;
$src_ids{$_} = [ ids($src{$_}) ] for @ids;
my $tot_src = 0; $tot_src += scalar @{ $src_ids{$_} } for @ids;
say sprintf("source identifiers: %d total, %.1f per thread", $tot_src, $tot_src / @ids);

my @arms = ( [ 'teacher (stored)', \%teacher ] );
for my $g (@gens) {
   open my $gfh, '<', $g or die "$g: $!";
   my (%s, $name);
   while (my $r = $csv->getline($gfh)) {
      next unless @$r >= 3;
      $name ||= $r->[1] || $g;
      $s{ $r->[0] } = b64($r->[2]);
   }
   close $gfh;
   push @arms, [ $name // $g, \%s ];
}

my @out;
for my $a (@arms) { push @out, score(@$a) }

say "";
printf "%-30s %6s %7s %7s %8s %9s %8s\n",
       'arm', 'ids', 'exact', 'drift', 'invented', 'ok prefix', 'covered';
say '-' x 82;
for my $r (@out) {
   printf "%-30s %6d %6.1f%% %6.1f%% %7.1f%% %8.1f %7.1f%%\n",
          $r->{label}, $r->{n},
          pc($r->{exact}, $r->{n}), pc($r->{drift}, $r->{n}), pc($r->{inv}, $r->{n}),
          $r->{drift} ? $r->{prefix_sum} / $r->{drift} : 0,
          pc($r->{covered}, $tot_src);
}
say '-' x 82;
say "ids       = identifier-shaped strings found in that arm's summaries";
say "exact     = identical to one in the source thread";
say "drift     = shared the first >= $MIN_PREFIX characters with a source identifier, then differed";
say "invented  = no source identifier shared even that much";
say "ok prefix = for DRIFT only, mean characters correct before it went wrong";
say "covered   = share of SOURCE identifiers the summaries mentioned at all (exactly)";

for my $r (@out) {
   next unless @{ $r->{worst} };
   say "";
   say "$r->{label} — worst drifts (source -> summary, correct prefix marked with |):";
   my @w = (sort { $b->[2] <=> $a->[2] } @{ $r->{worst} })[ 0 .. ($SHOW - 1 < $#{ $r->{worst} } ? $SHOW - 1 : $#{ $r->{worst} }) ];
   for my $d (@w) {
      my ($source, $got, $plen) = @$d;
      say sprintf("  %-26s -> %s|%s",
                  $source, substr($got, 0, $plen), substr($got, $plen));
   }
}

# ---------------------------------------------------------------------------
sub score {
   my ($label, $sum) = @_;
   my ($n, $exact, $drift, $inv, $psum) = (0) x 5;
   my (@worst, %covered);
   for my $id (@ids) {
      my $s = $sum->{$id};
      next unless defined $s && length $s;
      my @have = ids($s);
      my @pool = @{ $src_ids{$id} };
      my %pool_u = map { uc($_) => $_ } @pool;
      for my $g (@have) {
         $n++;
         my $gu = uc $g;
         if (exists $pool_u{$gu}) { $exact++; $covered{"$id\0$gu"} = 1; next }
         my ($best, $blen) = best_prefix_match($g, \@pool);
         if ($best && $blen >= $MIN_PREFIX) {
            $drift++; $psum += $blen;
            push @worst, [ $best, $g, $blen ];
         } else {
            $inv++;
         }
      }
   }
   return { label => $label, n => $n, exact => $exact, drift => $drift, inv => $inv,
            prefix_sum => $psum, worst => \@worst, covered => scalar keys %covered };
}

sub ids { return identifiers($_[0], min_len => $MIN_LEN, dates => $DATES) }

sub pc { my ($x, $t) = @_; return $t ? 100 * $x / $t : 0 }
sub b64 { my $x = decode_base64(shift); utf8::decode($x); return $x }

sub usage {
   return "usage: $0 pairs.csv [gen1.csv ...] [--min-prefix 4] [--min-len 5] " .
          "[--show 12] [--dates]\n";
}
