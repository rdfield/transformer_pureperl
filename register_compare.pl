#!/usr/bin/perl
# Compare the REGISTER of candidate training corpora against a reference.
#
# WHY THIS EXISTS
#   The client target corpus is formal NON-NATIVE business English. Measured
#   2026-10-05: "kindly" appears 16.7 times per 10k words in client and 0.0 times
#   in Enron, with 34 contractions per 10k against Enron's 92. Those are the
#   features a summariser actually learns -- politeness formulae, sentence
#   length, contraction use -- so they are what should decide which corpora go
#   into a pretrain blend. Domain similarity is a much weaker signal: Enron is
#   real business email and still the wrong register.
#
#   The whole 50 GB cosmo pretrain corpus is Mixtral-generated textbook prose,
#   so it is doubly mismatched -- synthetic AND the wrong register. Before
#   spending days downloading and re-encoding a replacement, measure it.
#
# USAGE
#   perl register_compare.pl --ref client=corpus.txt  cosmo=shard.csv:text  ...
#
#   Each argument is LABEL=PATH[:FORMAT]. The reference (--ref) is what the
#   others are scored against; without one you just get the raw table.
#
#   FORMAT       meaning
#     (omitted)  plain text, one document per line or free-running
#     N          CSV, column number N (0-based)
#     name       CSV, column with that header name
#     b64:N      CSV, column N holding base64-encoded text
#     b64:name   ditto by header name
#     b64line    plain file, one base64-encoded document per line
#   .gz is handled transparently.
#
#   --bytes N    stop after N bytes per corpus (default 50e6). Register
#                markers converge quickly; a 50 MB sample of a 30 GB shard set
#                gives the same answer as the whole thing and takes seconds.
#   --min-words  refuse to report a corpus with fewer words than this (default
#                20000), because a rate per 10k words is meaningless below it.
#
use strict;
use warnings;
use Getopt::Long;
use Text::CSV;
use MIME::Base64 qw(decode_base64);

my $REF;
my $MAX_BYTES = 50_000_000;
my $MIN_WORDS = 20_000;
my $VERBOSE   = 0;
GetOptions('ref=s' => \$REF, 'bytes=i' => \$MAX_BYTES,
           'min-words=i' => \$MIN_WORDS, 'verbose' => \$VERBOSE)
   or die usage();

# ---------------------------------------------------------------------------
# The markers. Each is a rate per 10k words unless noted.
#
# Chosen to separate formal non-native business English from both native
# informal email and from encyclopaedic/textbook prose. Grouped by what they
# detect, because a blend can match on one axis and miss badly on another.
# ---------------------------------------------------------------------------
my @MARKERS = (
   # --- politeness formulae: the strongest non-native business signal -------
   { key => 'kindly',    label => 'kindly',        re => qr/\bkindly\b/i },
   { key => 'please',    label => 'please',        re => qr/\bplease\b/i },
   { key => 'revert',    label => 'revert(=reply)',re => qr/\brevert\b/i },
   { key => 'needful',   label => 'needful',       re => qr/\bneedful\b/i },
   { key => 'advise',    label => 'advise',        re => qr/\badvise\b/i },
   { key => 'await',     label => 'await(ing)',    re => qr/\bawait(?:ing|s)?\b/i },
   { key => 'thesame',   label => '"the same" obj',re => qr/\b(?:find|do|send|confirm|arrange|process)\s+the\s+same\b/i },
   { key => 'grateful',  label => 'grateful/appreciate if',
                                                  re => qr/\b(?:grateful|appreciate\s+if|appreciated\s+if)\b/i },
   # --- native informality: high in Enron, low in client, near-zero in textbooks
   { key => 'contract',  label => 'contractions',  re => qr/\b\w+'(?:s|t|re|ve|ll|d|m)\b/i },
   # --- email STRUCTURE: present in real mail, absent from web/textbook prose
   { key => 'quoted',    label => 'quoted-reply markers',
                                                  re => qr/(?:^|\s)(?:>+\s|-{2,}\s*Original Message|From:\s|Sent:\s|wrote:)/m },
   { key => 'sigoff',    label => 'sign-offs',     re => qr/\b(?:best regards|kind regards|regards|many thanks|br,)\b/i },
   # --- identifiers: the project's measured failure mode lives here ---------
   { key => 'longdigit', label => 'digit runs >=4',re => qr/(?<!\d)\d{4,}(?!\d)/ },
   { key => 'alnumref',  label => 'ALNUM refs',    re => qr/\b[A-Z]{2,}[-\/ ]?\d{3,}\b/ },
   # --- textbook/encyclopaedic tells: high in cosmo, low in email -----------
   { key => 'however',   label => 'however/moreover/furthermore',
                                                  re => qr/\b(?:however|moreover|furthermore|nevertheless)\b/i },
   { key => 'inconc',    label => 'in conclusion/overall',
                                                  re => qr/\b(?:in conclusion|overall,|in summary)\b/i },
);

# ---------------------------------------------------------------------------
sub open_corpus {
   my ($path) = @_;
   if ($path =~ /\.gz$/) {
      open(my $fh, '-|', 'gzip', '-dc', $path) or die "gzip $path: $!";
      return $fh;
   }
   open(my $fh, '<:encoding(UTF-8)', $path) or die "$path: $!";
   return $fh;
}

# Pull up to $MAX_BYTES of TEXT out of a corpus, whatever its container.
sub read_text {
   my ($path, $format) = @_;
   my $fh = open_corpus($path);
   my $text = '';
   my $bytes = 0;

   if (!defined $format || $format eq '') {              # plain text
      while (my $l = <$fh>) {
         $text .= $l; $bytes += length $l;
         last if $bytes >= $MAX_BYTES;
      }
   } elsif ($format eq 'b64line') {
      # A plain file with one base64-encoded document per line, which is how
      # this project stores the Enron corpus (see base64_to_text.pl).
      while (my $l = <$fh>) {
         chomp $l; next unless length $l;
         my $d = decode_base64($l); utf8::decode($d);
         $text .= "$d\n"; $bytes += length($d) + 1;
         last if $bytes >= $MAX_BYTES;
      }
   } else {
      my ($b64, $col) = $format =~ /^b64:(.+)$/ ? (1, $1) : (0, $format);
      my $csv = Text::CSV->new({ binary => 1, allow_loose_quotes => 1 });
      my $hdr = $csv->getline($fh) or die "$path: no header row\n";
      my $idx;
      if ($col =~ /^\d+$/) { $idx = $col }
      else {
         for my $i (0 .. $#$hdr) { $idx = $i if ($hdr->[$i] // '') eq $col }
         die "$path: no column named '$col' (have: @$hdr)\n" unless defined $idx;
      }
      while (my $row = $csv->getline($fh)) {
         my $cell = $row->[$idx] // next;
         if ($b64) { $cell = decode_base64($cell); utf8::decode($cell) }
         $text .= "$cell\n"; $bytes += length($cell) + 1;
         last if $bytes >= $MAX_BYTES;
      }
   }
   close $fh;
   return $text;
}

sub measure {
   my ($text) = @_;
   my %m;
   # Words: runs of letters/digits/apostrophes. Deliberately crude and
   # identical for every corpus, because the rates are only ever compared with
   # each other -- a different tokenisation would change all of them together.
   my @words = $text =~ /([\p{L}\p{N}'][\p{L}\p{N}'-]*)/g;
   my $w = scalar @words;
   return { words => 0 } unless $w;

   # Sentences: terminal punctuation followed by space/EOL, plus a line-break
   # fallback so bullet lists and headers are not counted as one vast sentence.
   my $sent = () = $text =~ /[.!?](?:\s|$)/g;
   $sent = 1 if $sent < 1;

   for my $mk (@MARKERS) {
      my $c = () = $text =~ /$mk->{re}/g;
      $m{$mk->{key}} = 10_000 * $c / $w;
   }
   $m{words}      = $w;
   $m{wps}        = $w / $sent;
   $m{wordlen}    = (length(join '', @words)) / $w;
   my %uniq; $uniq{lc $_}++ for @words;
   $m{ttr}        = 1000 * scalar(keys %uniq) / $w;   # types per 1000 tokens
   my $caps = () = $text =~ /\b[A-Z]{3,}\b/g;
   $m{allcaps}    = 10_000 * $caps / $w;
   return \%m;
}

# ---------------------------------------------------------------------------
my @specs;
push @specs, "REF\t$REF" if defined $REF;
push @specs, "\t$_"      for @ARGV;
die usage() unless @specs;

my (@labels, %data, $ref_label);
for my $spec (@specs) {
   my ($is_ref, $arg) = split /\t/, $spec, 2;
   my ($label, $rest) = $arg =~ /^([^=]+)=(.+)$/
      or die "bad spec '$arg' -- want LABEL=PATH[:FORMAT]\n";
   # Split PATH:FORMAT from the right, so absolute paths survive.
   my ($path, $format) = ($rest, '');
   if ($rest =~ /^(.*):((?:b64:)?[^:\/]+)$/ && -r $1) { ($path, $format) = ($1, $2) }
   die "cannot read $path\n" unless -r $path;

   printf STDERR "reading %-10s %s%s\n", $label, $path,
          ($format ? " [$format]" : '') if $VERBOSE;
   my $m = measure(read_text($path, $format));
   if ($m->{words} < $MIN_WORDS) {
      printf STDERR "SKIP %s: only %d words (need %d)\n", $label, $m->{words}, $MIN_WORDS;
      next;
   }
   push @labels, $label;
   $data{$label} = $m;
   $ref_label = $label if $is_ref eq 'REF';
}
die "nothing measured\n" unless @labels;

# ---------------------------------------------------------------------------
my @rows = (
   (map { [ $_->{label}, $_->{key} ] } @MARKERS),
   [ 'words/sentence',    'wps'     ],
   [ 'mean word length',  'wordlen' ],
   [ 'types per 1k',      'ttr'     ],
   [ 'ALLCAPS runs',      'allcaps' ],
);

my $w1 = 34;
printf "%-${w1}s", 'marker (per 10k words)';
printf "%12s", $_ for @labels;
print "\n", '-' x ($w1 + 12 * @labels), "\n";
for my $r (@rows) {
   my ($label, $key) = @$r;
   printf "%-${w1}s", $label;
   printf "%12.2f", $data{$_}{$key} for @labels;
   print "\n";
}
printf "%-${w1}s", 'words sampled';
printf "%12s", fmt_n($data{$_}{words}) for @labels;
print "\n";

# ---------------------------------------------------------------------------
# Distance to the reference: mean absolute RELATIVE difference across the
# markers. Relative rather than z-scored because with a handful of corpora a
# z-score is dominated by whichever outlier happens to be present, and because
# "40% off on average" is a number you can argue with.
#
# Markers where BOTH are ~0 are skipped rather than counted as a perfect match:
# two corpora that never say "needful" agree on nothing of interest.
if (defined $ref_label) {
   print "\n";
   printf "%-${w1}s", "mean |rel. diff| vs $ref_label";
   for my $l (@labels) {
      if ($l eq $ref_label) { printf "%12s", '--'; next }
      my ($sum, $n) = (0, 0);
      for my $r (@rows) {
         my $k = $r->[1];
         my ($a, $b) = ($data{$ref_label}{$k}, $data{$l}{$k});
         next if $a < 0.05 && $b < 0.05;        # both absent: no information
         my $denom = $a > 0 ? $a : $b;
         $sum += abs($a - $b) / $denom; $n++;
      }
      printf "%12.2f", $n ? $sum / $n : 0;
   }
   print "\n";
   print "\nLower is closer to $ref_label. Read the ROWS too, not just this\n",
         "number: a corpus can match on politeness and miss entirely on email\n",
         "structure, and those are different things to fix.\n";
}

sub fmt_n { my $n = shift; 1 while $n =~ s/^(\d+)(\d{3})/$1,$2/; return $n }

sub usage {
   return <<"USAGE";
usage: $0 [--ref LABEL=PATH[:FORMAT]] LABEL=PATH[:FORMAT] ...

  FORMAT: omitted = plain text; N or name = CSV column; b64:N or b64:name =
          base64-encoded CSV column; b64line = plain file of base64 lines.
          .gz handled transparently.
  --bytes N      sample size per corpus (default 50000000)
  --min-words N  refuse to report below this many words (default 20000)
  --verbose      say what is being read

example:
  $0 --ref client=/path/corpus_16000_2.txt \\
     cosmo=/path/train-00000-of-00139.parquet.csv:text \\
     enron=/path/enron_filtered_8000.txt
USAGE
}
