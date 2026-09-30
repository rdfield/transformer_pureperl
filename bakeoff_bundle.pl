#!/usr/bin/perl
# Stage 2 of the teacher bake-off: build a BLIND grading bundle.
#
# Two design rules, both learned the hard way on this project:
#
#  1. GRADE AGAINST THE SOURCE THREAD, never candidate-against-incumbent. Scoring
#     against a machine-generated reference gave 3-10% where grading against the
#     thread gave 1 A / 19 B / 0 C on the same model — the reference-based score
#     collapsed "wrong detail" into "wrong". So the thread is printed in full above
#     the candidates, and no candidate is ever labelled as the reference.
#
#  2. BLIND, with a per-thread randomised order. The incumbent summaries are the
#     ones you already trust; knowing which is which would flatter them. Labels are
#     1..N, the mapping goes to a separate key file, and the order is reshuffled
#     for every thread so position carries no information either.
#
# Grade each summary A / B / C against the thread above it, using the same rubric
# as the 2026-09-23 audit so the numbers stay comparable:
#   A = faithful and acceptable to send
#   B = fluent and plausible but a detail is wrong
#   C = garbled, repetitive, or unusable
#
#   perl bakeoff_bundle.pl sample.csv gen1.csv [gen2.csv ...] \
#        --out grade.txt --key key.csv [--seed 42] [--include-stored]
#
use Modern::Perl;
use Getopt::Long;
use Text::CSV;
use MIME::Base64 qw(decode_base64);

my $OUT  = '';
my $KEY  = '';
my $SEED = 42;
my $INCLUDE_STORED = 0;
GetOptions('out=s' => \$OUT, 'key=s' => \$KEY, 'seed=i' => \$SEED,
           'include-stored' => \$INCLUDE_STORED) or die usage();

my $sample = shift @ARGV or die usage();
my @gens = @ARGV;
die usage() unless @gens && $OUT && $KEY;

my $csv = Text::CSV->new({ binary => 1 });
open my $sfh, '<', $sample or die "$sample: $!";
my (@ids, %band, %body, %stored);
while (my $r = $csv->getline($sfh)) {
   next unless @$r >= 4;
   push @ids, $r->[0];
   $band{ $r->[0] }   = $r->[1];
   $body{ $r->[0] }   = b64($r->[2]);
   $stored{ $r->[0] } = b64($r->[3]);
}
close $sfh;

my %cand;           # id -> { label => summary }
my @labels;
if ($INCLUDE_STORED) {
   push @labels, 'stored(old prompt)';
   $cand{$_}{'stored(old prompt)'} = $stored{$_} for @ids;
}
for my $g (@gens) {
   open my $gfh, '<', $g or die "$g: $!";
   my $name;
   while (my $r = $csv->getline($gfh)) {
      next unless @$r >= 3;
      $name ||= $r->[1] || $g;
      $cand{ $r->[0] }{$name} = b64($r->[2]);
   }
   close $gfh;
   push @labels, $name if defined $name;
}
die "no candidates loaded\n" unless @labels;

srand($SEED);
open my $ofh, '>:encoding(UTF-8)', $OUT or die "$OUT: $!";
open my $kfh, '>', $KEY or die "$KEY: $!";
my $kcsv = Text::CSV->new({ binary => 1, eol => "\n" });
$kcsv->print($kfh, [ 'thread_id', 'label', 'model' ]);

print $ofh <<"HDR";
TEACHER BAKE-OFF — BLIND GRADING SHEET
$OUT   (seed $SEED, @{[scalar @labels]} candidates, @{[scalar @ids]} threads)

Grade every summary A / B / C against the THREAD printed above it.
  A = faithful and acceptable to send
  B = fluent and plausible, but a detail is wrong
  C = garbled, repetitive, or unusable

Do NOT try to work out which model wrote which. The mapping is in the key file
and the order is reshuffled per thread, so position tells you nothing.

Record grades in a plain file, one line per thread:
    <thread_id> 1=A 2=B 3=A
then: perl bakeoff_tally.pl $KEY grades.txt

HDR

my $t = 0;
for my $id (@ids) {
   $t++;
   my @have = grep { defined $cand{$id}{$_} && length $cand{$id}{$_} } @labels;
   next unless @have;
   # Fisher-Yates per thread: position must not leak the model
   for (my $i = $#have; $i > 0; $i--) {
      my $j = int rand($i + 1);
      @have[ $i, $j ] = @have[ $j, $i ];
   }

   print $ofh '=' x 78, "\n";
   printf $ofh "THREAD %d of %d   id %s   band %s   %d chars\n",
           $t, scalar @ids, $id, $band{$id}, length $body{$id};
   print $ofh '=' x 78, "\n";
   print $ofh $body{$id}, "\n\n";
   print $ofh '-' x 78, "\n";
   for my $k (1 .. @have) {
      my $s = $cand{$id}{ $have[ $k - 1 ] };
      $s =~ s/\s+\z//;
      printf $ofh "%d) %s\n\n", $k, $s;
      $kcsv->print($kfh, [ $id, $k, $have[ $k - 1 ] ]);
   }
   printf $ofh "GRADES %s: %s\n\n\n", $id,
           join(' ', map { "$_=_" } 1 .. scalar @have);
}
close $ofh; close $kfh;

say "";
say "candidates: " . join(', ', @labels);
say "threads:    " . scalar @ids;
say "wrote $OUT  (grade this)";
say "wrote $KEY  (do not open until grading is finished)";

sub b64 { my $x = decode_base64(shift); utf8::decode($x); return $x }

sub usage {
   return "usage: $0 sample.csv gen1.csv [gen2.csv ...] --out grade.txt " .
          "--key key.csv [--seed 42] [--include-stored]\n";
}
