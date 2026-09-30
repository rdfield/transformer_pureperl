#!/usr/bin/perl
# Stage 1.5 of the teacher bake-off: an AUTOMATIC screen, run before any human
# grading, so reading time is only spent on candidates that clear a mechanical bar.
#
# Human grading is the scarce resource here — 100 threads is an hour of careful
# reading — and some failures do not need eyes at all. A teacher that invents
# names is disqualified by measurement.
#
# The headline metric is RARE-TOKEN COPY RECALL, computed the same way the trainer
# computes it so the numbers are comparable with everything already recorded:
# encode source and summary with the shared BPE, take summary ids >= --rare-min
# (6000 by default, the same threshold as --copy_rare_min), and ask what share of
# them appear anywhere in the source. Its complement is the INVENTION RATE. For
# scale, the measured figures on this project are 37.5% recall for the broken
# student, 69.2% for the same architecture on denoising.
#
# Also reported, because each is a distinct way to be a bad teacher:
#   length vs the incumbent   — a teacher that writes 3x longer changes the task
#   repeated 3-grams          — degenerate repetition, the failure this project
#                               has already fought (KhKhKh, table rules)
#   refusal rate              — "cannot summarise" using the production phrase list
#   truncation               — finish_reason of 'length' means max_tokens was hit
#                               and the summary is cut off mid-sentence
#
#   perl bakeoff_screen.pl vocab.model sample.csv gen1.csv [gen2.csv ...]
#        [--rare-min 6000] [--ngram 3]
#
use Modern::Perl;
use Getopt::Long;
use lib '.';
use Text::CSV;
use MIME::Base64 qw(decode_base64);
use ML::Tokeniser;

my $RARE_MIN = 6000;
my $NGRAM    = 3;
GetOptions('rare-min=i' => \$RARE_MIN, 'ngram=i' => \$NGRAM) or die usage();

my $model  = shift @ARGV or die usage();
my $sample = shift @ARGV or die usage();
my @gens   = @ARGV;
die usage() unless @gens;
die "vocab $model not found\n"  unless -e $model;
die "sample $sample not found\n" unless -e $sample;

my $REFUSAL = qr/(no meaningful summary|cannot be summarized meaningfully|
   lacks sufficient information to summarize|it cannot be summarized|
   contains no meaningful content|a meaningful summary cannot be provided|
   it cannot be meaningfully summari[sz]ed|cannot summarise email|
   cannot be summarised|unable to summari[sz]e)/xi;

my $tok = ML::Tokeniser->new(vocab_size => 1000);
$tok->load(file => $model);
say "vocab loaded: $model";

# ---- sample: source body + the stored (incumbent) summary ----
my $csv = Text::CSV->new({ binary => 1 });
open my $sfh, '<', $sample or die "$sample: $!";
my (%body, %stored, @ids);
while (my $r = $csv->getline($sfh)) {
   next unless @$r >= 4;
   my ($id) = $r->[0];
   push @ids, $id;
   $body{$id}   = b64($r->[2]);
   $stored{$id} = b64($r->[3]);
}
close $sfh;
say "sample: " . scalar(@ids) . " threads";

# source id sets, encoded once and reused across every candidate
my %src_ids;
for my $id (@ids) {
   my %set = map { $_ => 1 } @{ $tok->encode(text => $body{$id}) };
   $src_ids{$id} = \%set;
}

my @report;
push @report, score('stored (old prompt)', { map { $_ => $stored{$_} } @ids }, {});

for my $g (@gens) {
   open my $gfh, '<', $g or die "$g: $!";
   my (%sum, %fin, $label);
   while (my $r = $csv->getline($gfh)) {
      next unless @$r >= 3;
      $label ||= $r->[1];
      $sum{ $r->[0] } = b64($r->[2]);
      $fin{ $r->[0] } = $r->[6] // '';
   }
   close $gfh;
   push @report, score($label // $g, \%sum, \%fin);
}

# ---- report ----
say "";
printf "%-34s %5s %7s %7s %7s %6s %6s %6s\n",
       'candidate', 'n', 'recall', 'invent', 'chars', 'rep3', 'refus', 'trunc';
say '-' x 88;
for my $r (@report) {
   printf "%-34s %5d %6.1f%% %6.1f%% %7.0f %5.1f%% %5.1f%% %5.1f%%\n",
          $r->{label}, $r->{n}, 100 * $r->{recall}, 100 * (1 - $r->{recall}),
          $r->{chars}, 100 * $r->{rep}, 100 * $r->{refuse}, 100 * $r->{trunc};
}
say '-' x 88;
say "recall = share of RARE summary tokens (id >= $RARE_MIN) present in the source.";
say "invent = its complement: tokens the summary asserts that the source never said.";
say "rep3   = share of summaries with a repeated $NGRAM-gram.  trunc = hit max_tokens.";
say "";
my $base = $report[0];
for my $r (@report[1 .. $#report]) {
   my @flag;
   push @flag, sprintf('invention %+.1fpp vs stored', 100 * ($base->{recall} - $r->{recall}))
      if $r->{recall} < $base->{recall} - 0.05;
   push @flag, sprintf('%.0f%% truncated — raise --max-tokens and regenerate', 100 * $r->{trunc})
      if $r->{trunc} > 0.02;
   # relative to the incumbent, not absolute: 7% of real gpt-4.1-mini summaries
   # contain a repeated 3-gram, which is just ordinary English.
   push @flag, sprintf('repetition %.0f%% vs %.0f%% for stored',
                       100 * $r->{rep}, 100 * $base->{rep})
      if $r->{rep} > $base->{rep} + 0.10;
   push @flag, sprintf('%.1fx the length of stored', $r->{chars} / ($base->{chars} || 1))
      if $base->{chars} && ($r->{chars} / $base->{chars} > 1.75 || $r->{chars} / $base->{chars} < 0.5);
   say @flag ? "FLAG $r->{label}: " . join('; ', @flag)
             : "ok   $r->{label}: clears the screen, send to grading";
}

sub score {
   my ($label, $sum, $fin) = @_;
   my ($n, $rare_tot, $rare_hit, $chars, $rep, $refuse, $trunc) = (0) x 7;
   for my $id (@ids) {
      my $s = $sum->{$id};
      next unless defined $s && length $s;
      $n++;
      $chars += length $s;
      $refuse++ if $s =~ $REFUSAL;
      $trunc++  if ($fin->{$id} // '') eq 'length';
      my $sids = $tok->encode(text => $s);
      for my $t (@$sids) {
         next unless $t >= $RARE_MIN;
         $rare_tot++;
         $rare_hit++ if $src_ids{$id}{$t};
      }
      # repeated n-gram anywhere in the summary (word level, case-insensitive)
      my @w = map { lc } ($s =~ /(\w+)/g);
      if (@w >= $NGRAM * 2) {
         my %seen;
         for my $i (0 .. $#w - $NGRAM + 1) {
            my $k = join ' ', @w[ $i .. $i + $NGRAM - 1 ];
            if ($seen{$k}++) { $rep++; last }
         }
      }
   }
   return {
      label  => $label,
      n      => $n,
      recall => $rare_tot ? $rare_hit / $rare_tot : 0,
      chars  => $n ? $chars / $n : 0,
      rep    => $n ? $rep / $n : 0,
      refuse => $n ? $refuse / $n : 0,
      trunc  => $n ? $trunc / $n : 0,
   };
}

sub b64 { my $t = decode_base64(shift); utf8::decode($t); return $t }

sub usage {
   return "usage: $0 vocab.model sample.csv gen1.csv [gen2.csv ...] " .
          "[--rare-min 6000] [--ngram 3]\n";
}
