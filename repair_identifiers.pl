#!/usr/bin/perl
# Repair drifted identifiers in generated summaries, as a post-processing pass.
#
# THE FAILURE THIS FIXES (measured, see measure_identifiers.pl):
#     source   PO1234567890123
#     summary  PO123456789654321     right for 9 digits, then drifts
# Teacher summaries are 96.8% exact on identifiers; the model is 27.9%, with a
# mean of 6.9 correct characters before it goes wrong. So the model reliably
# decides THAT an identifier should be copied and starts in the RIGHT PLACE — it
# just cannot hold the copy across the 4-6 BPE tokens a long reference occupies,
# and partial credit is worthless: PO123456789654321 is as useless as a blank.
#
# Because the opening run is usually right, the source identifier can normally be
# recovered by prefix. That makes this a text-level fix needing no retraining and
# no change to decoding.
#
# IT ONLY ACTS WHEN THE ANSWER IS UNAMBIGUOUS. A repair is made when exactly ONE
# source identifier shares >= --min-prefix opening characters with the emitted
# one. Where several tie — the corpus genuinely contains near-identical refs, e.g.
# three FTQ- quote numbers that a 5-character prefix cannot separate — the pass
# DECLINES and leaves the text alone. Guessing there would turn a visibly wrong
# identifier into an invisibly wrong one, which is worse.
#
# Identifiers already exactly matching the source are never touched.
#
# --min-prefix 4 IS A MEASURED DEFAULT, not a guess. Swept on 100 real generations
# against real source threads:
#   min-prefix   repairs   exact vs source   agrees with teacher
#   none           0            55.3%              44.4%
#   5             11            65.9%              48.7%
#   4 (default)   14            68.9%              51.9%
#   3             19            73.3%              54.8%
# The second column alone CANNOT validate a repair — measure_identifiers.pl scores
# "exact" as matching ANY source identifier, so substituting the WRONG one still
# counts as success. The teacher-agreement column is the check that cannot be
# fooled, and it climbs too, so the gain is real rather than the metric flattering
# itself. 3 scores higher on both but its absolute agreement count stops rising
# (40 at both 4 and 3) while the denominator falls, so 4 is the point where both
# curves are still clearly climbing. Lower it to 3 deliberately if you want reach
# over caution.
#
# Input/output use the bake-off formats:
#   pairs.csv  id,band,body_b64,summary_b64   (bakeoff_sample.pl — supplies SOURCE)
#   gen.csv    id,model,summary_b64,...       (bakeoff_generate.pl / a generation dump)
#   out.csv    same shape as gen.csv, summaries repaired
# So the effect is measured with:
#   perl measure_identifiers.pl pairs.csv gen.csv out.csv
#
#   perl repair_identifiers.pl pairs.csv gen.csv out.csv
#        [--min-prefix 5] [--min-len 5] [--show 15] [--dry-run]
#
use Modern::Perl;
use Getopt::Long;
use lib '.';
use Text::CSV;
use MIME::Base64 qw(encode_base64 decode_base64);
use ML::Identifiers qw(identifiers best_prefix_match common_prefix);

my $MIN_PREFIX = 4;    # shortest opening run we will act on — swept, see below
my $MIN_LEN    = 5;
my $SHOW       = 15;
my $DRY        = 0;
GetOptions('min-prefix=i' => \$MIN_PREFIX, 'min-len=i' => \$MIN_LEN,
           'show=i' => \$SHOW, 'dry-run' => \$DRY) or die usage();

my $pairs = shift @ARGV or die usage();
my $gen   = shift @ARGV or die usage();
my $out   = $DRY ? undef : (shift @ARGV or die usage());
die "$pairs not found\n" unless -e $pairs;
die "$gen not found\n"   unless -e $gen;
die "refusing to overwrite input\n" if defined $out && ($out eq $gen || $out eq $pairs);

my $csv = Text::CSV->new({ binary => 1 });
open my $pfh, '<', $pairs or die "$pairs: $!";
my %src;
while (my $r = $csv->getline($pfh)) {
   next unless @$r >= 3;
   $src{ $r->[0] } = [ identifiers(b64($r->[2]), min_len => $MIN_LEN) ];
}
close $pfh;
say "source threads: " . scalar(keys %src);

my $ocsv = Text::CSV->new({ binary => 1, eol => "\n" });
my $ofh;
unless ($DRY) { open $ofh, '>', $out or die "$out: $!" }
open my $gfh, '<', $gen or die "$gen: $!";

my ($rows, $n_ids, $exact, $fixed, $ambig, $nomatch, $changed_rows) = (0) x 7;
my @log;
while (my $r = $csv->getline($gfh)) {
   next unless @$r >= 3;
   $rows++;
   my $id = $r->[0];
   my $s  = b64($r->[2]);
   my $pool = $src{$id} // [];
   my %pool_u = map { uc($_) => 1 } @$pool;

   my $before = $s;
   for my $got (identifiers($s, min_len => $MIN_LEN)) {
      $n_ids++;
      if ($pool_u{ uc $got }) { $exact++; next }          # already right
      my ($best, $plen, $tied) = best_prefix_match($got, $pool);
      if (!defined $best || $plen < $MIN_PREFIX) { $nomatch++; next }
      if ($tied > 1) {
         $ambig++;
         push @log, [ 'AMBIG', $got, $best, $plen, $tied ];
         next;
      }
      # substitute every occurrence, case-insensitively, keeping the source's casing
      my $q = quotemeta $got;
      my $hits = ($s =~ s/$q/$best/gi);
      next unless $hits;
      $fixed++;
      push @log, [ 'FIX', $got, $best, $plen, 1 ];
   }
   $changed_rows++ if $s ne $before;

   unless ($DRY) {
      my @row = @$r;
      my $b = $s; utf8::encode($b);
      $row[2] = encode_base64($b, '');
      $ocsv->print($ofh, \@row);
   }
}
close $gfh;
close $ofh unless $DRY;

say "";
say sprintf("summaries            %6d  (%d rewritten)", $rows, $changed_rows);
say sprintf("identifiers seen     %6d", $n_ids);
say sprintf("  already exact      %6d  %5.1f%%", $exact,   pc($exact));
say sprintf("  REPAIRED           %6d  %5.1f%%", $fixed,   pc($fixed));
say sprintf("  declined, ambiguous%6d  %5.1f%%  (>=2 source ids tie on the prefix)", $ambig, pc($ambig));
say sprintf("  no prefix match    %6d  %5.1f%%  (<%d chars shared — invented, not drifted)",
            $nomatch, pc($nomatch), $MIN_PREFIX);

if (@log) {
   say "";
   say "sample decisions (| marks the shared opening run):";
   my @s = (sort { $b->[3] <=> $a->[3] } @log)[ 0 .. ($SHOW - 1 < $#log ? $SHOW - 1 : $#log) ];
   for my $d (@s) {
      my ($what, $got, $best, $plen, $tied) = @$d;
      say sprintf("  %-5s %s|%s  ->  %s%s", $what,
                  substr($got, 0, $plen), substr($got, $plen), $best,
                  $tied > 1 ? "   ($tied candidates tie — left alone)" : '');
   }
}
say $DRY ? "\n(dry run — nothing written)" : "\nwrote $out";

sub pc { my $x = shift; return $n_ids ? 100 * $x / $n_ids : 0 }
sub b64 { my $t = decode_base64(shift); utf8::decode($t); return $t }

sub usage {
   return "usage: $0 pairs.csv gen.csv out.csv [--min-prefix 5] [--min-len 5] " .
          "[--show 15] [--dry-run]\n";
}
