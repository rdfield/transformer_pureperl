use Modern::Perl;
use lib '.';
use ML::Tokeniser;
my $tok = ML::Tokeniser->new(vocab_size => 10);
my $filename = "client_emails.model";

$tok->load(file => "$filename");

my $string = 'zab test asdqwrgverf jhrth gfdggrge';
my @after  = $tok->encode(text => $string);

my $text = $tok->decode(ids => \@after);
say $text;
if ($text eq $string) {
  say "all good";
} else {
  say "no good";
}


