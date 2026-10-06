#!/usr/bin/perl
# Turn generated summaries into stage-2 training pairs: filter the bad ones, then
# encode to the trainer's format.
#
#   in:   sample.csv  id,band,body_b64,""        (bakeoff_sample*.pl)
#         gen.csv     id,model,summary_b64,...   (bakeoff_generate.pl)
#   out:  pairs.txt   email_ids,16003,summary_ids   one per line
#
# EVERY FILTER HERE IS A MEASURED ONE, from grading 40 Qwen summaries of Enron
# email (17A/2B/1C for flash, 15A/3B/2C for the 235B):
#
#  1. REFUSALS — 38-40% of Enron gets "cannot summarise email". That is correct
#     behaviour: the production prompt is built for transactional "who needs what
#     from whom" requests and much of Enron is chatter, newsletters and banter.
#     But 50k pairs containing 20k refusals would teach the model to REFUSE,
#     which is worse than the hallucination this whole exercise is fixing.
#     (The refusal regex uses \s+ rather than literal spaces: under /x, literal
#     spaces are stripped FROM THE PATTERN, which silently broke this check once
#     already and hid the 40% rate completely.)
#
#  2. TOO SHORT — the worst single output in the grading was "Firstname-A x00000", a
#     12-character signature fragment from the 235B. The threshold is 25, NOT the
#     123 that the shortest A-graded summary suggested: sorting all 122
#     non-refusal summaries by length shows a clean gap at 12 -> 34, and
#     everything from 34 up is a real sentence ("Firstname-B will dig up TVA for
#     Firstname-D", 34; "Firstname-C attached the Management Report for Feb. 23, 2000", 56).
#     Setting it from the shortest OBSERVED good summary rather than from where
#     the garbage actually stops discarded ~5% of valid terse summaries and would
#     have biased stage 2 toward verbosity — the wrong direction, since more
#     words means more chances for a wrong detail.
#
#  3. NAME DUMPS — one C-graded summary listed 14 cc recipients instead of
#     summarising. Counting multi-word capitalised runs separates it cleanly: no
#     A-graded summary had more than 6, that one had 12.
#
#  4. TRUNCATED — finish_reason 'length' means the summary was cut off mid
#     sentence. Training on half a sentence teaches half sentences.
#
#  5. SOURCE OVER THE TOKEN BUDGET — the summary was generated from the FULL
#     email, but training truncates the source at --email_budget. Any pair whose
#     source exceeds that budget teaches the model that facts may appear in a
#     summary without being present in the input: precisely the hallucination
#     being cured. This MUST be counted in tokens, not bytes. Capping the
#     sampler at 8000 bytes was supposed to keep sources under 2048 tokens on a
#     ~4.24 bytes/token average, but dense BOM listings tokenise nearer 1.8
#     bytes/token and 9.3% of pairs came out truncated (max 4,361 tokens). It
#     surfaced only in the trainer's startup line. A byte proxy is not a token
#     budget.
#
# After these, the residual C rate in the graded sample was ~1 in 40 (2.5%),
# which is ordinary training-set noise.
#
# The pair format must match what MVEncDecTrainer expects for fine-tuning:
# source ids, then the [SEP] id, then summary ids. Source is NOT truncated here —
# --email_budget does that at training time, consistently with validation.
#
#   perl build_stage2_pairs.pl vocab.model sample.csv gen.csv pairs.txt
#        [--min-chars 100] [--max-names 8] [--report-only]
#
use Modern::Perl;
use Getopt::Long;
use lib '.';
use Text::CSV;
use MIME::Base64 qw(decode_base64);
use ML::Tokeniser;

my $MIN_CHARS = 25;    # see the note below — 100 was wrong
my $MAX_SRC   = 2048;  # source token budget; see note 5
my $MAX_NAMES = 8;
my $SEP       = 16003;
my $REPORT    = 0;
GetOptions('min-chars=i' => \$MIN_CHARS, 'max-names=i' => \$MAX_NAMES,
           'max-src-tokens=i' => \$MAX_SRC,
           'sep=i' => \$SEP, 'report-only' => \$REPORT) or die usage();

my $model  = shift @ARGV or die usage();
my $sample = shift @ARGV or die usage();
my $gen    = shift @ARGV or die usage();
my $out    = $REPORT ? undef : (shift @ARGV or die usage());
die "$model not found\n"  unless -e $model;
die "$sample not found\n" unless -e $sample;
die "$gen not found\n"    unless -e $gen;

my $REFUSAL = qr/(cannot \s+ summarise \s+ email|cannot \s+ be \s+ summari[sz]ed|
   no \s+ meaningful \s+ summary|lacks \s+ sufficient \s+ (?:information|context)|
   unable \s+ to \s+ summari[sz]e|contains \s+ no \s+ meaningful \s+ content|
   a \s+ meaningful \s+ summary \s+ cannot \s+ be \s+ provided|
   not \s+ enough \s+ (?:information|context))/xi;

my $csv = Text::CSV->new({ binary => 1 });
open my $sf, '<', $sample or die "$sample: $!";
my %src;
while (my $r = $csv->getline($sf)) { next unless @$r >= 3; $src{ $r->[0] } = $r->[2] }
close $sf;

my $tok = ML::Tokeniser->new(vocab_size => 16000);
$tok->load(file => $model);
say "vocab loaded; sources: " . scalar(keys %src);

open my $gf, '<', $gen or die "$gen: $!";
my $ofh;
unless ($REPORT) { open $ofh, '>', $out or die "$out: $!" }

my ($n, $kept, $n_ref, $n_short, $n_names, $n_trunc, $n_nosrc, $n_long) = (0) x 8;
my ($tok_src, $tok_sum) = (0, 0);
my @examples;
while (my $r = $csv->getline($gf)) {
   next unless @$r >= 3;
   $n++;
   my $id = $r->[0];
   unless (exists $src{$id}) { $n_nosrc++; next }
   my $s = b64($r->[2]);
   $s =~ s/\A\s+|\s+\z//g;

   if (($r->[6] // '') eq 'length') { $n_trunc++; next }
   if ($s =~ $REFUSAL)              { $n_ref++;   next }
   if (length($s) < $MIN_CHARS)     { $n_short++; push @examples, ['SHORT', $s]; next }
   my $names = () = $s =~ /\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+/g;
   if ($names >= $MAX_NAMES)        { $n_names++; push @examples, ['NAMES', $s]; next }

   my $body = b64($src{$id});
   my $sids = $tok->encode(text => $body);
   my $tids = $tok->encode(text => $s);
   next unless @$sids && @$tids;
   if ($MAX_SRC && @$sids > $MAX_SRC) { $n_long++; next; }
   $kept++; $tok_src += scalar @$sids; $tok_sum += scalar @$tids;
   print $ofh join(',', @$sids) . ",$SEP," . join(',', @$tids) . "\n" unless $REPORT;
}
close $gf;
close $ofh unless $REPORT;

my $pc = sub { $n ? sprintf('%5.1f%%', 100 * $_[0] / $n) : '  -  ' };
say "";
say "generated summaries   $n";
say "dropped:";
say sprintf("  refusal         %6d  %s", $n_ref,   $pc->($n_ref));
say sprintf("  truncated       %6d  %s  (finish_reason length)", $n_trunc, $pc->($n_trunc));
say sprintf("  under %3d chars %6d  %s", $MIN_CHARS, $n_short, $pc->($n_short));
say sprintf("  >=%d names      %6d  %s  (recipient-list dumps)", $MAX_NAMES, $n_names, $pc->($n_names));
say sprintf("  src >%d tok   %6d  %s  (summary was made from text the model cannot see)",
            $MAX_SRC, $n_long, $pc->($n_long)) if $MAX_SRC;
say sprintf("  no source       %6d  %s", $n_nosrc, $pc->($n_nosrc)) if $n_nosrc;
say "kept:";
say sprintf("  %6d pairs  %s", $kept, $pc->($kept));
say sprintf("  mean %.0f source tokens, %.0f summary tokens", $kept ? $tok_src/$kept : 0,
            $kept ? $tok_sum/$kept : 0) if $kept;
if (@examples) {
   say "\nsample of what was dropped:";
   for my $e (@examples[0 .. ($#examples < 4 ? $#examples : 4)]) {
      say sprintf("  %-6s %s", $e->[0], substr($e->[1], 0, 90));
   }
}
say $REPORT ? "\n(report only — nothing written)" : "\nwrote $out";

sub b64 { my $t = decode_base64(shift); utf8::decode($t); return $t }
sub usage {
   return "usage: $0 vocab.model sample.csv gen.csv pairs.txt " .
          "[--min-chars 25] [--max-names 8] [--max-src-tokens 2048] [--report-only]\n";
}
