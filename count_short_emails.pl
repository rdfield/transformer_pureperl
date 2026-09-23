use Modern::Perl;
use Text::CSV;
use MIME::Base64 qw(decode_base64);
use utf8;
use lib '.';
use ML::Tokeniser;
use Data::Dumper;

my $filename = shift @ARGV;

die "no file to process" unless defined($filename) and -e $filename;
my $csv = Text::CSV->new();
open my $fhin, "<:encoding(utf8)", $filename or die $!;

my $model = "client_emails_16000.model";

my $tok = ML::Tokeniser->new(vocab_size => 276);
$tok->load(file => "$model");

my $sep_id = $tok->{special_tokens}{"[SEP]"};
die "no [SEP] defined in model" unless $sep_id;

my $processed = 0;
my $few_lines = 0; # 2 lines or less
my $short_text = 0; # 160 chars or less
my $definitely_short = 0; # 2 lines or less and 160 chars or less
while (my $row = $csv->getline($fhin)) {
   my $body = decode_base64($row->[1]);
   utf8::decode($body);
   my $count = () = $body =~ /\n/g;
   my $lastchar = substr $body, -1,1;
   $count += ($lastchar ne "\n");
my $print_line = 0;
   do { $print_line = 1; $few_lines++} if $count <= 2;
   do { $print_line = 1;$short_text++} if length($body) <= 160;
   do { $print_line = 1;$definitely_short++ } if (length($body) <= 160 and $count <= 2);
say $body if $print_line;
   $processed++;
}

say "$processed processed";
say "$definitely_short definitely short, $short_text short text, $few_lines few lines";
