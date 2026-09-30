#!/usr/bin/perl
# Stage 0 of the teacher bake-off: draw a LENGTH-STRATIFIED sample of threads
# that already have a teacher summary, so every candidate teacher is judged on
# identical inputs.
#
# WHY STRATIFIED AND NOT UNIFORM: measured on email_sumararies_regen_v4.txt
# (12,793 pairs) the body length distribution is
#   p10 102  p25 295  median 701  p75 1360  p90 2380  p99 7570  max 88,481 chars
# A uniform draw therefore lands mostly around 700 characters and would contain
# almost no long threads — and long threads are exactly where teachers diverge,
# because a weak model drops the middle of a long chain and invents a plausible
# conclusion for it. Stratifying forces the hard cases into the sample.
#
# Excluded: bodies under --min-chars (grading "cannot be summarised" against an
# empty body measures nothing) and existing teacher REFUSALS (76 of 12,793 =
# 0.6%, detected with the same phrase list the production script uses).
#
# Output is one CSV row per thread: id,band,body_b64,summary_b64
# base64 because bodies are multi-line; the same file feeds every candidate.
#
#   perl bakeoff_sample.pl email_sumararies_regen_v4.txt sample.csv [--n 100]
#        [--seed 42] [--min-chars 200] [--bands 500,1400,3000]
#
use Modern::Perl;
use Getopt::Long;
use Text::CSV;
use MIME::Base64 qw(encode_base64 decode_base64);

my $N         = 100;
my $SEED      = 42;
my $MIN_CHARS = 200;
my $BANDS     = '500,1400,3000';
GetOptions('n=i' => \$N, 'seed=i' => \$SEED, 'min-chars=i' => \$MIN_CHARS,
           'bands=s' => \$BANDS) or die usage();

my $in  = shift @ARGV or die usage();
my $out = shift @ARGV or die usage();
die "input $in not found\n" unless -e $in;
die "refusing to overwrite input\n" if $out eq $in;

# Same refusal phrases the production script screens on, so a thread the
# incumbent teacher already gave up on never reaches a grader.
my $REFUSAL = qr/(no meaningful summary|cannot be summarized meaningfully|
   lacks sufficient information to summarize|lacks meaningful information to summarize|
   it cannot be summarized|contains no meaningful content|
   a meaningful summary cannot be provided|it cannot be meaningfully summari[sz]ed|
   lacks sufficient context or details|lacks enough context or content to summarize|
   without any clear request or specific details|
   lacks any meaningful content or context to summarize|
   no clear request or business details to summarize|cannot summarise email|
   is unclear and does not contain enough information|
   content is missing or incomplete|is unclear and lacks sufficient information)/xi;

my @cuts = sort { $a <=> $b } split /,/, $BANDS;
my $nbands = @cuts + 1;
my $band_of = sub {
   my $len = shift;
   for my $i (0 .. $#cuts) { return $i if $len < $cuts[$i] }
   return scalar @cuts;
};
my @band_label = do {
   my @l; my $lo = $MIN_CHARS;
   for my $c (@cuts) { push @l, "$lo-" . ($c - 1); $lo = $c }
   push @l, "$lo+";
   @l;
};

my $csv = Text::CSV->new({ binary => 1 }) or die Text::CSV->error_diag;
open my $fh, '<', $in or die "$in: $!";

my @pool;                       # [ band ][ rows ]
my ($rows, $n_short, $n_refuse, $n_bad) = (0) x 4;
while (my $r = $csv->getline($fh)) {
   next unless @$r >= 3;
   $rows++;
   my ($body, $summ);
   eval {
      $body = decode_base64($r->[1]); utf8::decode($body);
      $summ = decode_base64($r->[2]); utf8::decode($summ);
      1;
   } or do { $n_bad++; next };
   if (length($body) < $MIN_CHARS) { $n_short++; next }
   if ($summ =~ $REFUSAL)          { $n_refuse++; next }
   push @{ $pool[ $band_of->(length $body) ] }, [ $r->[0], $body, $summ ];
}
close $fh;

# Deterministic Fisher-Yates so a given --seed always yields the same sample.
srand($SEED);
my $shuffle = sub {
   my $a = shift;
   for (my $i = $#$a; $i > 0; $i--) {
      my $j = int rand($i + 1);
      @$a[$i, $j] = @$a[$j, $i];
   }
};

my $per = int($N / $nbands);
my @take;
my @shortfall;
for my $b (0 .. $nbands - 1) {
   my $avail = $pool[$b] // [];
   $shuffle->($avail);
   my $want = $per;
   push @shortfall, "$band_label[$b] wanted $want has " . scalar(@$avail)
      if @$avail < $want;
   push @take, [ $b, @$_ ] for @{$avail}[ 0 .. ($want - 1 < $#$avail ? $want - 1 : $#$avail) ];
}
# top up from the largest remaining bands if a band ran dry, so --n is honoured
if (@take < $N) {
   my %used = map { $_->[1] => 1 } @take;
   for my $b (reverse 0 .. $nbands - 1) {
      for my $row (@{ $pool[$b] // [] }) {
         last if @take >= $N;
         next if $used{ $row->[0] }++;
         push @take, [ $b, @$row ];
      }
      last if @take >= $N;
   }
}

open my $ofh, '>', $out or die "$out: $!";
my $ocsv = Text::CSV->new({ binary => 1, eol => "\n" });
my @per_band = (0) x $nbands;
for my $t (@take) {
   my ($band, $id, $body, $summ) = @$t;
   $per_band[$band]++;
   my ($b, $s) = ($body, $summ);
   utf8::encode($b); utf8::encode($s);
   $ocsv->print($ofh, [ $id, $band_label[$band],
                        encode_base64($b, ''), encode_base64($s, '') ]);
}
close $ofh;

say "";
say "input:  $in  ($rows rows)";
say "excluded:  $n_short under $MIN_CHARS chars, $n_refuse teacher refusals"
    . ($n_bad ? ", $n_bad undecodable" : "");
say "eligible per band:";
say sprintf("  %-12s %6d available", $band_label[$_], scalar @{ $pool[$_] // [] })
   for 0 .. $nbands - 1;
say "sampled (seed $SEED):";
say sprintf("  %-12s %6d", $band_label[$_], $per_band[$_]) for 0 .. $nbands - 1;
say sprintf("  %-12s %6d total", 'ALL', scalar @take);
say "WARNING: band short of target — $_" for @shortfall;
say "\nwrote $out";

sub usage {
   return "usage: $0 pairs.csv sample.csv [--n 100] [--seed 42] " .
          "[--min-chars 200] [--bands 500,1400,3000]\n";
}
