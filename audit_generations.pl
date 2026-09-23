#!/usr/bin/perl
# Audit MVEncDecInfer / MVSLMInfer output for MODE COLLAPSE and LOSS OF
# CONDITIONING -- the failure where generations are fluent, correctly-lengthed
# and almost entirely INDEPENDENT of the input.
#
# Teacher-forced token accuracy CANNOT see this: it feeds the correct prefix,
# so it scores the decoder's language model over summary style rather than the
# conditioning. Run this on the generations instead.
#
#   cd embedding_cpp && ./build/MVEncDecInfer --model ... --n 40 \
#       | perl /home/mvine/embedding/audit_generations.pl
#
#   perl audit_generations.pl saved_run.txt --ngram 4 --shifts 5
#
# Reads GEN/REF token-id lines; needs no vocab (it works on ids). Pipe through
# decode_gen.pl separately when you want to read the text.
use Modern::Perl;
use Getopt::Long;
use List::Util qw(sum min);

# ---------------------------------------------------------------------------
# PROVISIONAL bars. Derived from ONE measurement: the d=256/16K thread run of
# 2026-09-17, which scored diversity 0.24 and conditioning 0.08 while its own
# references scored 1.00 by construction. Re-derive these once a healthy run
# exists -- do not treat them as established.
# ---------------------------------------------------------------------------
my $BAR_DIVERSITY   = 0.60;   # gen n-gram novelty as a fraction of the refs'
my $BAR_PREFIX      = 0.60;   # gen distinct-prefix rate as a fraction of refs'
my $BAR_CONDITION   = 0.15;   # share of available headroom explained by input

my $NGRAM  = 4;
my $SHIFTS = 5;               # average the mismatched-reference score over N shifts
my @PREFIX = (5, 10, 20);
GetOptions('ngram=i' => \$NGRAM, 'shifts=i' => \$SHIFTS)
   or die "usage: $0 [file] [--ngram N] [--shifts N]\n";

my (@gen, @ref);
while (my $line = <>) {
   if ($line =~ /^(GEN|REF)\s*(.*)$/) {
      my @ids = grep { /^\d+$/ } split /,/, $2;
      push @{ $1 eq 'GEN' ? \@gen : \@ref }, \@ids;
   }
}
die "no GEN lines found on input\n" unless @gen;
warn sprintf("warning: %d GEN vs %d REF lines; using %d\n", scalar @gen, scalar @ref, min(scalar @gen, scalar @ref)) if @ref && @gen != @ref;
my $n = @ref ? min(scalar @gen, scalar @ref) : scalar @gen;

# --- per-set structural stats -------------------------------------------------
sub stats {
   my ($seqs, $label) = @_;
   return undef unless @$seqs;
   my %s = (label => $label, n => scalar @$seqs);

   for my $k (@PREFIX) {
      my %seen;
      $seen{ join ',', @{$_}[0 .. min($k, scalar @$_) - 1] }++ for @$seqs;
      $s{"prefix$k"} = scalar keys %seen;
   }

   my (%all, $tot, $rep) = ((), 0, 0);
   for my $seq (@$seqs) {
      my %here;
      for my $i (0 .. $#$seq - $NGRAM + 1) {
         my $g = join ',', @{$seq}[$i .. $i + $NGRAM - 1];
         $all{$g}++; $here{$g}++; $tot++;
      }
      $rep += sum(map { $_ - 1 } values %here) // 0;
   }
   $s{ngrams}    = $tot;
   $s{distinct}  = $tot ? (scalar keys %all) / $tot : 0;   # cross-example novelty
   $s{selfrep}   = $tot ? $rep / $tot : 0;                 # within-example looping
   $s{len}       = (sum map { scalar @$_ } @$seqs) / @$seqs;

   # the single most-reused opening, and how many examples share it
   my %open;
   $open{ join ',', @{$_}[0 .. min($PREFIX[0], scalar @$_) - 1] }++ for @$seqs;
   ($s{top_open}) = sort { $open{$b} <=> $open{$a} || $a cmp $b } keys %open;
   $s{top_open_n} = $open{ $s{top_open} };
   return \%s;
}

my $g = stats(\@gen, 'generated');
my $r = @ref ? stats(\@ref, 'reference') : undef;

# --- conditioning: own reference vs deliberately WRONG references --------------
# Unigram precision of each generation against its own reference, and against
# other examples' references. If those are equal the output ignores its input.
my ($own, @shifted);
if (@ref) {
   my @refset = map { my %h; $h{$_} = 1 for @$_; \%h } @ref;
   my $score = sub {
      my ($shift) = @_;
      my $acc = 0;
      for my $i (0 .. $n - 1) {
         my $seq = $gen[$i];
         next unless @$seq;
         my $set = $refset[ ($i + $shift) % $n ];
         $acc += (grep { $set->{$_} } @$seq) / @$seq;
      }
      return $acc / $n;
   };
   $own = $score->(0);
   push @shifted, $score->($_) for 1 .. min($SHIFTS, $n - 1);
}
my $wrong  = @shifted ? (sum @shifted) / @shifted : undef;
# share of the achievable headroom that reading the input actually explains
my $signal = defined $wrong && $wrong < 1 ? ($own - $wrong) / (1 - $wrong) : undef;

# --- report -------------------------------------------------------------------
printf "\n%d generations, %d references, %d-grams, %d shift(s)\n\n",
       scalar @gen, scalar @ref, $NGRAM, scalar @shifted;

printf "%-28s %12s %12s\n", '', 'GENERATED', 'REFERENCE';
printf "%-28s %12s %12s\n", '-' x 28, '-' x 12, '-' x 12;
for my $k (@PREFIX) {
   printf "%-28s %12s %12s\n", "distinct $k-token prefixes",
          "$g->{\"prefix$k\"}/$g->{n}",
          $r ? "$r->{\"prefix$k\"}/$r->{n}" : '-';
}
printf "%-28s %12.3f %12s\n", "distinct $NGRAM-grams",   $g->{distinct},
       $r ? sprintf('%.3f', $r->{distinct}) : '-';
printf "%-28s %12.3f %12s\n", "within-seq repeats",      $g->{selfrep},
       $r ? sprintf('%.3f', $r->{selfrep}) : '-';
printf "%-28s %12.1f %12s\n", "mean length (tokens)",    $g->{len},
       $r ? sprintf('%.1f', $r->{len}) : '-';
print "\n";

printf "most-reused opening: %d/%d generations share [%s]\n",
       $g->{top_open_n}, $g->{n}, $g->{top_open};
printf "  (references: %d/%d share their commonest opening)\n",
       $r->{top_open_n}, $r->{n} if $r;

if (defined $signal) {
   print "\nCONDITIONING (unigram precision of each generation)\n";
   printf "  against its OWN reference    %.3f\n", $own;
   printf "  against WRONG references     %.3f   (mean of %d shifts)\n",
          $wrong, scalar @shifted;
   printf "  input signal                 %+.3f  = %.1f%% of available headroom\n",
          $own - $wrong, 100 * $signal;
}

# --- verdict ------------------------------------------------------------------
print "\n";
my @fail;
if ($r) {
   my $div = $r->{distinct}  ? $g->{distinct} / $r->{distinct} : 1;
   my $pre = $r->{"prefix$PREFIX[0]"}
           ? ($g->{"prefix$PREFIX[0]"} / $g->{n}) / ($r->{"prefix$PREFIX[0]"} / $r->{n}) : 1;
   printf "diversity   %.2f of reference  (bar %.2f)  %s\n",
          $div, $BAR_DIVERSITY, $div < $BAR_DIVERSITY ? 'FAIL' : 'ok';
   printf "opening var %.2f of reference  (bar %.2f)  %s\n",
          $pre, $BAR_PREFIX, $pre < $BAR_PREFIX ? 'FAIL' : 'ok';
   push @fail, 'diversity' if $div < $BAR_DIVERSITY;
   push @fail, 'opening variety' if $pre < $BAR_PREFIX;
}
if (defined $signal) {
   printf "conditioning %.2f              (bar %.2f)  %s\n",
          $signal, $BAR_CONDITION, $signal < $BAR_CONDITION ? 'FAIL' : 'ok';
   push @fail, 'conditioning' if $signal < $BAR_CONDITION;
}
printf "within-seq repeats %.3f -- %s\n", $g->{selfrep},
       $g->{selfrep} > 0.05 ? 'degenerate looping; try --no_repeat_ngram 3'
                            : 'no degenerate looping';

print "\n";
if (@fail) {
   say "VERDICT: FAIL (" . join(', ', @fail) . ")";
   say "  Low diversity + low conditioning together = MODE COLLAPSE: the model";
   say "  emits the corpus's commonest summary regardless of input. Decoding";
   say "  flags will not fix it -- they suppress repetition without adding";
   say "  content. The encoder is under-trained relative to the decoder.";
} else {
   say "VERDICT: PASS -- generations are diverse and track their inputs.";
}
exit(@fail ? 1 : 0);
