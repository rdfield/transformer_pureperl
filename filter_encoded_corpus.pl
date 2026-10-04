#!/usr/bin/perl
# Filter an encoded pretraining corpus (one doc per line, comma-separated BPE
# ids) before it goes into --pretrain.
#
# Removes two things a straight encode leaves behind:
#
#  1. BINARY documents. Base64 attachments and embedded images that survived
#     into the markdown decode to high-bit byte soup. A byte-level BPE cannot
#     merge binary, so it encodes at ~1 token per byte: maximum token cost,
#     zero linguistic content. Detected by the share of ids in the raw
#     single-byte range 128-255, which real text almost never needs once BPE
#     has merged it (measured: clean client docs sit near 0%, offenders 35-89%).
#
#  2. OUT-OF-RANGE ids. An id >= vocab_size indexes PAST the embedding table, so
#     the GPU does an out-of-bounds read and the process dies with an illegal
#     memory access — typically surfacing at whatever CUDA error check runs next,
#     which makes it look like a hardware fault. Measured 2026-10-04: FOUR such
#     ids in a 2.11-billion-token corpus (max 942549 against a vocab of 16005)
#     killed four consecutive pretrain runs at 18%, 20%, 44% and 64%. The shuffle
#     is reseeded per run, so the poisoned window lands somewhere different every
#     time, which is what disguised it as thermal instability for three days.
#     FOUR BAD TOKENS IN TWO BILLION. Always pass --vocab.
#
#  3. EXACT DUPLICATE documents — whole threads exported more than once. This
#     is corpus-level duplication, distinct from the within-thread line
#     duplication already handled upstream in the HTML->markdown step.
#
# Long documents are KEPT. load_pretrain_windows() concatenates everything into
# one stream and slices fixed windows across it, so a long thread becomes many
# windows rather than one truncated example -- truncating here would just throw
# tokens away.
#
#   perl filter_encoded_corpus.pl in.txt out.txt [--hi 0.30] [--report-only]
#
use Modern::Perl;
use Getopt::Long;
use Digest::MD5 qw(md5_hex);

my $VOCAB       = 0;      # vocab size; ids >= this are out of range (0 = skip the check)
my $HI_FRAC     = 0.30;   # drop docs with more than this share of 128-255 ids
my $MIN_TOKENS  = 2;      # a 1-token doc cannot form a window
my $REPORT_ONLY = 0;
GetOptions('hi=f' => \$HI_FRAC, 'min=i' => \$MIN_TOKENS, 'report-only' => \$REPORT_ONLY,
           'vocab=i' => \$VOCAB)
   or die "usage: $0 in.txt out.txt [--vocab 16005] [--hi 0.30] [--min 2] [--report-only]\n";
warn "WARNING: no --vocab given, so out-of-range ids will NOT be checked\n" unless $VOCAB;

my $in  = shift @ARGV or die "no input file\n";
my $out = $REPORT_ONLY ? undef : (shift @ARGV or die "no output file\n");
die "input $in not found\n" unless -e $in;
die "refusing to overwrite input\n" if defined $out && $out eq $in;

open my $fh, '<', $in or die "$in: $!";
my $ofh;
unless ($REPORT_ONLY) {
   open $ofh, '>', $out or die "$out: $!";
}

my %seen;
my ($docs, $kept, $tok_in, $tok_out) = (0, 0, 0, 0);
my ($n_bin, $n_dup, $n_short, $t_bin, $t_dup, $t_short) = (0) x 6;
my ($n_oor, $t_oor) = (0, 0);
my @worst_oor;
my @worst;      # largest binary docs, for the report
my $maxlen = 0;

while (my $line = <$fh>) {
   chomp $line;
   next unless $line =~ /\d/;
   $docs++;

   my @ids = grep { /^\d+$/ } split /,/, $line;
   my $n = scalar @ids;
   $tok_in += $n;

   if ($n < $MIN_TOKENS) { $n_short++; $t_short += $n; next; }

   # out-of-range ids: fatal at training time, so drop the document
   if ($VOCAB) {
      my @oor = grep { $_ >= $VOCAB } @ids;
      if (@oor) {
         $n_oor++; $t_oor += $n;
         push @worst_oor, [ $., scalar @oor, (sort { $b <=> $a } @oor)[0] ];
         next;
      }
   }

   # 1. binary
   my $hi = grep { $_ >= 128 && $_ <= 255 } @ids;
   my $frac = $hi / $n;
   if ($frac > $HI_FRAC) {
      $n_bin++; $t_bin += $n;
      push @worst, [ $., $n, $frac ];
      next;
   }

   # 2. exact duplicate (hash the id string, not the text — same thing here
   #    and far cheaper than decoding)
   my $h = md5_hex($line);
   if ($seen{$h}++) { $n_dup++; $t_dup += $n; next; }

   $kept++; $tok_out += $n;
   $maxlen = $n if $n > $maxlen;
   print $ofh "$line\n" unless $REPORT_ONLY;
}
close $fh;
close $ofh unless $REPORT_ONLY;

my $pc = sub { $tok_in ? sprintf('%5.2f%%', 100 * $_[0] / $tok_in) : '   -  ' };
say "";
say "input:  $in";
say sprintf("  %7d docs  %12s tokens", $docs, commify($tok_in));
say "dropped:";
say sprintf("  binary      %6d docs  %12s tokens  %s  (>%.0f%% ids in 128-255)",
            $n_bin, commify($t_bin), $pc->($t_bin), 100 * $HI_FRAC);
say sprintf("  duplicate   %6d docs  %12s tokens  %s", $n_dup, commify($t_dup), $pc->($t_dup));
say sprintf("  too short   %6d docs  %12s tokens  %s", $n_short, commify($t_short), $pc->($t_short));
say sprintf("  OUT OF RANGE %5d docs  %12s tokens  %s  (id >= %d — these CRASH training)",
            $n_oor, commify($t_oor), $pc->($t_oor), $VOCAB) if $VOCAB;
say "kept:";
say sprintf("  %7d docs  %12s tokens  %s   longest doc %s",
            $kept, commify($tok_out), $pc->($tok_out), commify($maxlen));
if (@worst) {
   say "largest binary documents dropped (line, tokens, high-byte share):";
   my @top = (sort { $b->[1] <=> $a->[1] } @worst)[0 .. ($#worst < 9 ? $#worst : 9)];
   say sprintf("  line %7d  %10s tok  %5.1f%%", $_->[0], commify($_->[1]), 100 * $_->[2]) for @top;
}
if (@worst_oor) {
   say "out-of-range documents dropped (line, count, largest id):";
   say sprintf("  line %7d  %4d bad  max id %d", @$_) for @worst_oor[0 .. ($#worst_oor < 9 ? $#worst_oor : 9)];
}
say $REPORT_ONLY ? "\n(report only — nothing written)" : "\nwrote $out";

sub commify { my $n = reverse shift; $n =~ s/(\d{3})(?=\d)/$1,/g; return scalar reverse $n }
