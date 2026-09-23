use Modern::Perl;
use lib '.';
use ML::Tokeniser;
use MIME::Base64 qw(decode_base64);

my $model = shift @ARGV;
my $infile = shift @ARGV;

die "no model name" unless $model and -e $model;
die "no input file" unless $infile and -e $infile;

my $tok = ML::Tokeniser->new(vocab_size => 1000);
$tok->load(file => $model);

open(my $fh, '<:encoding(UTF-8)', $infile) or die $!;

open FH, ">", "$infile.out" or die $!;
my $processed = 0;
while (my $line = <$fh>) {
   next unless $line =~ /\w/;
   chomp($line);
   $line = decode_base64($line);
   utf8::decode($line);
   my @ids = @{$tok->encode(text => $line)};
   print FH join(",", @ids) . "\n";

}
close FH;
