#!/usr/bin/perl
# Stage 0 sampler for the Enron teacher bake-off — and the same path the full
# 50k generation run uses, just with a bigger --n.
#
# WHY ENRON RATHER THAN A PUBLIC SUMMARISATION SET. Stage 2 of the curriculum has
# to teach reading-to-summarise, and the licence-clean public sets are nearly all
# LONG documents: govreport ~9k tokens loses 77% to a 2048 budget, billsum 49%,
# arXiv 66%. A truncated document whose summary describes content you cut away
# teaches the model that facts may appear in summaries without being present in
# the input — which is exactly the hallucination failure being fixed. Enron is
# email (exact domain match), median ~700 bytes so it FITS the budget, and public
# domain. It just has no summaries, which Qwen supplies for a few dollars.
#
# Input is enron_emails_base64.txt: one base64 email body per line, no summaries.
# Output is the bake-off pairs format with an EMPTY reference column, because
# there is no incumbent teacher here — the arms are compared against each other
# and against the source thread, not against a stored summary.
#
#   id,band,body_b64,""          <- empty 4th column
#
# Length-stratified for the same reason as bakeoff_sample.pl: a uniform draw of a
# median-960-char distribution would contain almost no long emails, and long ones
# are where teachers diverge.
#
#   perl bakeoff_sample_enron.pl enron_emails_base64.txt sample.csv
#        [--n 100] [--seed 42] [--min-bytes 300] [--max-bytes 8000]
#        [--bands 800,2000,4000]
#
use Modern::Perl;
use Getopt::Long;
use Text::CSV;
use MIME::Base64 qw(encode_base64 decode_base64);

my $N         = 100;
my $SEED      = 42;
my $MIN_BYTES = 300;     # below this there is nothing to summarise
my $MAX_BYTES = 8000;    # above this it will not fit a 2048-token budget
my $BANDS     = '800,2000,4000';
GetOptions('n=i' => \$N, 'seed=i' => \$SEED, 'min-bytes=i' => \$MIN_BYTES,
           'max-bytes=i' => \$MAX_BYTES, 'bands=s' => \$BANDS) or die usage();

my $in  = shift @ARGV or die usage();
my $out = shift @ARGV or die usage();
die "$in not found\n" unless -e $in;
die "refusing to overwrite input\n" if $out eq $in;

my @cuts = sort { $a <=> $b } split /,/, $BANDS;
my $nbands = @cuts + 1;
my @label = do { my @l; my $lo = $MIN_BYTES;
                 for my $c (@cuts) { push @l, "${lo}-" . ($c-1); $lo = $c }
                 push @l, "${lo}+"; @l };
my $band_of = sub { my $n = shift;
                    for my $i (0..$#cuts) { return $i if $n < $cuts[$i] } return scalar @cuts };

open my $fh, '<', $in or die "$in: $!";
my (@pool, $lines, $n_short, $n_long, $n_bad);
while (my $l = <$fh>) {
   chomp $l;
   $lines++;
   next unless length $l;
   my $text = eval { my $t = decode_base64($l); utf8::decode($t); $t };
   unless (defined $text) { $n_bad++; next }
   my $bytes = do { my $c = $text; utf8::encode($c); length $c };
   if ($bytes < $MIN_BYTES) { $n_short++; next }
   if ($bytes > $MAX_BYTES) { $n_long++;  next }
   push @{ $pool[ $band_of->($bytes) ] }, [ $lines, $text, $bytes ];
}
close $fh;

srand($SEED);
my $shuffle = sub { my $a = shift;
   for (my $i = $#$a; $i > 0; $i--) { my $j = int rand($i+1); @$a[$i,$j] = @$a[$j,$i] } };

my $per = int($N / $nbands);
my (@take, @short_of);
for my $b (0 .. $nbands-1) {
   my $avail = $pool[$b] // [];
   $shuffle->($avail);
   push @short_of, "$label[$b] wanted $per has " . scalar(@$avail) if @$avail < $per;
   push @take, [ $b, @$_ ] for @{$avail}[ 0 .. ($per-1 < $#$avail ? $per-1 : $#$avail) ];
}

my $csv = Text::CSV->new({ binary => 1, eol => "\n" });
open my $ofh, '>', $out or die "$out: $!";
my @per_band = (0) x $nbands;
for my $t (@take) {
   my ($band, $srcline, $text, $bytes) = @$t;
   $per_band[$band]++;
   my $b = $text; utf8::encode($b);
   # 4th column deliberately empty: no incumbent summary exists for Enron
   $csv->print($ofh, [ $srcline, $label[$band], encode_base64($b, ''), '' ]);
}
close $ofh;

say "";
say "input:  $in  ($lines lines)";
say "excluded: $n_short under $MIN_BYTES bytes, $n_long over $MAX_BYTES"
    . ($n_bad ? ", $n_bad undecodable" : "");
say "eligible per band:";
say sprintf("  %-12s %7d", $label[$_], scalar @{ $pool[$_] // [] }) for 0..$nbands-1;
say "sampled (seed $SEED):";
say sprintf("  %-12s %7d", $label[$_], $per_band[$_]) for 0..$nbands-1;
say sprintf("  %-12s %7d total", 'ALL', scalar @take);
say "WARNING: $_" for @short_of;
say "\nwrote $out   (4th column empty — no incumbent summary for Enron)";

sub usage {
   return "usage: $0 enron_base64.txt sample.csv [--n 100] [--seed 42] " .
          "[--min-bytes 300] [--max-bytes 8000] [--bands 800,2000,4000]\n";
}
