#!/usr/bin/perl
# LINE-level cleaning of a pretraining corpus, run BEFORE encoding.
#
# Sits between base64_encode_data.pl and encode_input.pl: reads one base64
# document per line, strips junk LINES from inside each document, writes the
# same format back out. --plain instead treats the input as a raw text stream and
# filters it line by line (no document assembly, no --min-chars); that is the mode
# the measurements below were taken in, so it doubles as the verification path.
#
# WHY THIS IS NOT filter_encoded_corpus.pl's JOB: that filter works on ids and
# drops WHOLE DOCUMENTS (binary soup, exact duplicates). The junk here lives on
# individual lines INSIDE otherwise good documents, so a document-level filter
# cannot see it and a byte-level BPE cannot merge it away.
#
# MEASURED on corpus_16000.txt (839 MB, 14,314,827 lines) 2026-09-30:
#
#   rule/divider lines   944,719 lines   76.5 MB   9.1% of bytes
#   bare HTML tag lines  560,584 lines    3.8 MB   0.5%
#   blank lines        1,982,593 lines    2.4 MB   0.3%
#   kept              10,826,931 lines  756.5 MB  90.2%   (198M -> 178M tokens)
#
# So ~20M tokens per 200M are lines like `| | | ----------- | |`. That is worse
# than wasted budget: it TEACHES the model to emit table rules, and degenerate
# repetition is a failure mode this project has already had to fight. Run this
# before encoding hundreds of chunks, not after.
#
# Inline markup (589,291 kept lines carry a tag mid-prose) is REPORTED but not
# stripped by default — removing tags from inside a prose line can mangle the
# text around them. --strip-inline opts in.
#
#   perl clean_corpus_lines.pl in.b64 out.b64 [--report-only] [--plain]
#                              [--min-chars 40] [--rule-len 4] [--strip-inline]
#
use Modern::Perl;
use Getopt::Long;
use MIME::Base64 qw(encode_base64 decode_base64);

my $MIN_CHARS    = 40;   # drop a document with less than this surviving
my $RULE_LEN     = 4;    # a line of >= this many punctuation/space chars is a rule
my $REPORT_ONLY  = 0;
my $PLAIN        = 0;
my $STRIP_INLINE = 0;
GetOptions(
   'min-chars=i'  => \$MIN_CHARS,
   'rule-len=i'   => \$RULE_LEN,
   'report-only'  => \$REPORT_ONLY,
   'plain'        => \$PLAIN,
   'strip-inline' => \$STRIP_INLINE,
) or die usage();

my $in  = shift @ARGV or die usage();
my $out = $REPORT_ONLY ? undef : (shift @ARGV or die usage());
die "input $in not found\n" unless -e $in;
die "refusing to overwrite input\n" if defined $out && $out eq $in;

open my $fh, '<', $in or die "$in: $!";
my $ofh;
unless ($REPORT_ONLY) {
   open $ofh, '>', $out or die "$out: $!";
}

# A "rule" line is punctuation and whitespace only: markdown table borders,
# horizontal rules, setext underlines, runs of >>>> or ####. Anything with a
# letter or digit in it survives, so `| a | b |` is kept and `| | | --- |` is not.
my $RULE = qr/^[\s|+\-=_*#>~`.:]{$RULE_LEN,}$/;
my $TAG  = qr{^\s*</?[a-zA-Z][^>]*>\s*$};

my ($docs, $kept_docs, $n_short_doc) = (0, 0, 0);
my ($l_in, $l_rule, $l_tag, $l_blank, $l_kept) = (0) x 5;
my ($b_in, $b_rule, $b_tag, $b_blank, $b_kept) = (0) x 5;
my $l_inline = 0;

# --plain is a STREAMING line filter: raw text has newlines inside documents, so
# one input line is a line, not a document. Document assembly and --min-chars
# only apply to the base64 form, where one line really is one document.
if ($PLAIN) {
   my $blank_run = 0;
   while (my $l = <$fh>) {
      chomp $l;
      $l_in++; $b_in += length($l) + 1;
      if ($l =~ /^\s*$/) {
         $l_blank++; $b_blank += length($l) + 1;
         next if $blank_run++;
         print $ofh "\n" unless $REPORT_ONLY;
         next;
      }
      $blank_run = 0;
      if ($l =~ $RULE) { $l_rule++; $b_rule += length($l) + 1; next; }
      if ($l =~ $TAG)  { $l_tag++;  $b_tag  += length($l) + 1; next; }
      $l_inline++ if $l =~ m{</?[a-zA-Z][^>]{0,80}>};
      $l =~ s{</?[a-zA-Z][^>]{0,80}>}{}g if $STRIP_INLINE;
      $l_kept++; $b_kept += length($l) + 1;
      print $ofh "$l\n" unless $REPORT_ONLY;
   }
} else {
   while (my $line = <$fh>) {
      chomp $line;
      next unless length $line;
      $docs++;

      my $text = decode_base64($line);
      utf8::decode($text);

      my @out;
      my $blank_run = 0;
      for my $l (split /\n/, $text, -1) {
         $l_in++; $b_in += length($l) + 1;
         if ($l =~ /^\s*$/) {
            $l_blank++; $b_blank += length($l) + 1;
            # collapse runs of blank lines to a single separator
            next if $blank_run++;
            push @out, '';
            next;
         }
         $blank_run = 0;
         if ($l =~ $RULE) { $l_rule++; $b_rule += length($l) + 1; next; }
         if ($l =~ $TAG)  { $l_tag++;  $b_tag  += length($l) + 1; next; }
         $l_inline++ if $l =~ m{</?[a-zA-Z][^>]{0,80}>};
         $l =~ s{</?[a-zA-Z][^>]{0,80}>}{}g if $STRIP_INLINE;
         $l_kept++; $b_kept += length($l) + 1;
         push @out, $l;
      }

      # trim leading/trailing blank separators the collapse may have left
      shift @out while @out && $out[0]  eq '';
      pop   @out while @out && $out[-1] eq '';
      my $cleaned = join "\n", @out;

      if (length($cleaned) < $MIN_CHARS) { $n_short_doc++; next; }
      $kept_docs++;
      next if $REPORT_ONLY;

      utf8::encode($cleaned);
      print $ofh encode_base64($cleaned, '') . "\n";
   }
}
close $fh;
close $ofh unless $REPORT_ONLY;

my $pc = sub { $b_in ? sprintf('%5.1f%%', 100 * $_[0] / $b_in) : '   -  ' };
say "";
say "input:  $in" . ($PLAIN ? "  (--plain)" : "  (base64 per line)");
say $PLAIN ? sprintf("  %s lines  %s bytes", commify($l_in), commify($b_in))
           : sprintf("  %s documents  %s lines  %s bytes",
                     commify($docs), commify($l_in), commify($b_in));
say "dropped lines:";
say sprintf("  rule/divider  %12s lines  %12s bytes  %s   (>=%d chars of punctuation only)",
            commify($l_rule), commify($b_rule), $pc->($b_rule), $RULE_LEN);
say sprintf("  bare HTML tag %12s lines  %12s bytes  %s",
            commify($l_tag), commify($b_tag), $pc->($b_tag));
say sprintf("  blank         %12s lines  %12s bytes  %s   (runs collapsed to one)",
            commify($l_blank), commify($b_blank), $pc->($b_blank));
say "kept:";
say sprintf("  %12s lines  %12s bytes  %s", commify($l_kept), commify($b_kept), $pc->($b_kept));
say sprintf("  %12s documents   (%s dropped: under %d chars after cleaning)",
            commify($kept_docs), commify($n_short_doc), $MIN_CHARS) unless $PLAIN;
say sprintf("  of the kept lines, %s contain inline markup%s",
            commify($l_inline), $STRIP_INLINE ? ' (STRIPPED)' : ' (left alone; --strip-inline to remove)');
say sprintf("  ~tokens %s -> %s  at 4.24 bytes/token",
            commify(int($b_in / 4.24)), commify(int($b_kept / 4.24)));
say $REPORT_ONLY ? "\n(report only — nothing written)" : "\nwrote $out";

sub commify { my $n = reverse shift; $n =~ s/(\d{3})(?=\d)/$1,/g; return scalar reverse $n }

sub usage {
   return "usage: $0 in.b64 out.b64 [--report-only] [--plain] " .
          "[--min-chars 40] [--rule-len 4] [--strip-inline]\n";
}
