#!/usr/bin/perl
# Decode MVSLMInfer output (GEN/REF token-id lines) back to text via the
# shared BPE. Pipe MVSLMInfer's stdout through this:
#   cd embedding_cpp && ./build/MVSLMInfer --model slm_v1.safetensors \
#       --input /home/mvine/embedding/slm_val.csv --n 20 | perl /home/mvine/embedding/decode_gen.pl
use Modern::Perl;
use lib '/home/mvine/embedding';
use ML::Tokeniser;
binmode(STDOUT, ':utf8');

my $t = ML::Tokeniser->new(vocab_size => 16000);
$t->load(file => '/home/mvine/embedding/client_emails_16000.model');

while (my $line = <STDIN>) {
   chomp $line;
   if ($line =~ /^(GEN|REF) ?(.*)$/) {
      my ($tag, $ids) = ($1, $2);
      my @i = grep { length && /^\d+$/ } split /,/, $ids;
      my $txt = @i ? $t->decode(ids => \@i) : '(empty)';
      $txt =~ s/\s+/ /g;                 # flatten whitespace for readability
      say "$tag: $txt";
   } else {
      say $line;                          # pass through the "=== example ===" headers
   }
}
