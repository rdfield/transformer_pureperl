#!/usr/bin/perl
# Stage 4 of the teacher bake-off: unblind, tally, and apply the decision rule.
#
# THE DECISION RULE IS FIXED HERE, IN THE SCRIPT, BEFORE YOU GRADE. That is the
# point of writing it down: a rule chosen after seeing the numbers is a
# rationalisation. Change it only deliberately, and before grading, not after.
#
#   local A-rate within  5pp of incumbent  -> no detectable difference: scale up
#   local A-rate  5-15pp worse             -> use it for ENRON ONLY (no labels
#                                             there, so anything beats nothing);
#                                             keep the incumbent for client
#   local A-rate >15pp worse, or any C      -> not usable as a teacher
#
# AND A RESOLUTION WARNING THE SCRIPT ENFORCES: at n=100 with an A-rate near 50%
# the standard error is about 5pp per arm, so the difference between two arms
# carries roughly 7pp of noise. A gap inside that band means "cannot distinguish",
# NOT "equivalent" — so the top row of the rule above is a statement about what
# this test can see, not proof of parity. The script prints the 95% interval for
# every arm and says explicitly when a gap is inside the noise.
#
# Grades file: one line per thread, blanks and #comments ignored.
#     534 1=A 2=B 3=A
#     535 1=B 2=B 3=C
#
#   perl bakeoff_tally.pl key.csv grades.txt [--incumbent PATTERN] [--n-min 1]
#
use Modern::Perl;
use Getopt::Long;
use Text::CSV;

my $INCUMBENT = 'gpt';      # substring identifying the incumbent arm
my $N_MIN     = 1;
GetOptions('incumbent=s' => \$INCUMBENT, 'n-min=i' => \$N_MIN) or die usage();

my $keyfile = shift @ARGV or die usage();
my $grades  = shift @ARGV or die usage();
die "$keyfile not found\n" unless -e $keyfile;
die "$grades not found\n"  unless -e $grades;

my $csv = Text::CSV->new({ binary => 1 });
open my $kfh, '<', $keyfile or die "$keyfile: $!";
my %map;                     # id -> label -> model
my $hdr = $csv->getline($kfh);
while (my $r = $csv->getline($kfh)) {
   next unless @$r >= 3;
   $map{ $r->[0] }{ $r->[1] } = $r->[2];
}
close $kfh;
die "key file has no rows\n" unless %map;

my (%tally, %seen_thread, @warn);
open my $gfh, '<', $grades or die "$grades: $!";
while (my $l = <$gfh>) {
   $l =~ s/#.*//;
   next unless $l =~ /\S/;
   my ($id, @pairs) = split ' ', $l;
   $id =~ s/^GRADES//i; $id =~ s/[:,]$//;
   unless (exists $map{$id}) { push @warn, "grades mention unknown thread $id"; next }
   $seen_thread{$id}++;
   for my $p (@pairs) {
      my ($lab, $g) = $p =~ /^(\d+)\s*=\s*([ABCabc])$/ or do {
         push @warn, "thread $id: cannot parse '$p'"; next;
      };
      my $model = $map{$id}{$lab};
      unless (defined $model) { push @warn, "thread $id: no label $lab in key"; next }
      $tally{$model}{ uc $g }++;
      $tally{$model}{N}++;
   }
}
close $gfh;

my $ungraded = grep { !$seen_thread{$_} } keys %map;
push @warn, "$ungraded of " . scalar(keys %map) . " threads have no grades" if $ungraded;

# ---- report ----
my @models = sort {
   rate($b) <=> rate($a)
} grep { ($tally{$_}{N} // 0) >= $N_MIN } keys %tally;
die "nothing to tally\n" unless @models;

say "";
printf "%-34s %5s %5s %5s %5s %8s %-16s\n", 'arm', 'n', 'A', 'B', 'C', 'A-rate', '95% interval';
say '-' x 86;
for my $m (@models) {
   my $t = $tally{$m};
   my ($n, $a) = ($t->{N}, $t->{A} // 0);
   my $p  = $n ? $a / $n : 0;
   my $se = $n ? sqrt($p * (1 - $p) / $n) : 0;
   printf "%-34s %5d %5d %5d %5d %7.1f%%  %4.1f%% - %4.1f%%\n",
          $m, $n, $a, $t->{B} // 0, $t->{C} // 0, 100 * $p,
          100 * ($p - 1.96 * $se < 0 ? 0 : $p - 1.96 * $se),
          100 * ($p + 1.96 * $se > 1 ? 1 : $p + 1.96 * $se);
}
say '-' x 86;

my ($inc) = grep { /\Q$INCUMBENT\E/i } @models;
unless ($inc) {
   say "no arm matches --incumbent '$INCUMBENT', so no verdict.";
   say "arms present: " . join(', ', @models);
   finish();
}
say "incumbent: $inc";
say "";

my $ip = rate($inc);
for my $m (@models) {
   next if $m eq $inc;
   my $d  = 100 * ($ip - rate($m));                 # pp the candidate is behind
   my $c  = $tally{$m}{C} // 0;
   my $n  = $tally{$m}{N};
   my $ni = $tally{$inc}{N};
   # noise on the DIFFERENCE of two proportions
   my $se = 100 * sqrt( $ip * (1 - $ip) / ($ni || 1) + rate($m) * (1 - rate($m)) / ($n || 1) );
   my $band = 1.96 * $se;

   my $verdict;
   if ($c) {
      $verdict = "NOT USABLE — $c graded C. Garbled output as a teacher poisons the student.";
   } elsif ($d > 15) {
      $verdict = sprintf("NOT USABLE — %.1fpp behind the incumbent.", $d);
   } elsif ($d > 5) {
      $verdict = sprintf("ENRON ONLY — %.1fpp behind. Keep the incumbent for client.", $d);
   } else {
      $verdict = sprintf("SCALE UP — %.1fpp behind, inside the 5pp bar.", $d);
   }
   printf "%-34s %s\n", $m, $verdict;
   printf "%-34s   gap %+.1fpp +/- %.1fpp%s\n", '', -$d, $band,
      abs($d) < $band
        ? "  <-- INSIDE THE NOISE: cannot distinguish from the incumbent, which is"
        : '';
   say sprintf("%-34s   %s", '', "not the same as proving them equal. More threads would be needed.")
      if abs($d) < $band;
}

finish();

sub rate {
   my $t = $tally{ $_[0] } or return 0;
   return $t->{N} ? ($t->{A} // 0) / $t->{N} : 0;
}

sub finish {
   if (@warn) {
      say "";
      say "warnings:";
      say "  $_" for @warn;
   }
   exit 0;
}

sub usage {
   return "usage: $0 key.csv grades.txt [--incumbent gpt] [--n-min 1]\n";
}
