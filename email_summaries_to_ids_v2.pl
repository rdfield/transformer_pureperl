use Modern::Perl;
use Text::CSV;
use MIME::Base64 qw(decode_base64);
use utf8;
use lib '.';
use ML::Tokeniser;
use Data::Dumper;

my $model_file = shift @ARGV;
my $input_file = shift @ARGV;
my $output_file = shift @ARGV;

die "input file not found" unless -e $input_file;
die "model file not found" unless -e $model_file;
die "output file not specified" unless defined($output_file);
die "output file already exists" if -e $output_file;

my $csv = Text::CSV->new();
open my $fhin, "<:encoding(utf8)", $input_file or die $!;
open my $fhout, ">:encoding(utf8)", $output_file or die $!;

my $tok = ML::Tokeniser->new(vocab_size => 276);
$tok->load(file => $model_file);

my $sep_id = $tok->{special_tokens}{"[SEP]"};
die "no [SEP] defined in model" unless $sep_id;

my $processed = 0;
while (my $row = $csv->getline($fhin)) {
# email id expected at $row->[0];
   my $body = decode_base64($row->[1]);
   utf8::decode($body);
   my $summary = decode_base64($row->[2]);
   utf8::decode($summary);
   print $fhout  join(",", @{$tok->encode(text => $body)}) . ",$sep_id," . join(",", @{$tok->encode(text => $summary)}) . "\n";
   $processed++;
}
close $fhout;

say "$processed processed";
