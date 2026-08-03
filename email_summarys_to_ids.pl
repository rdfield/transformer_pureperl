use Modern::Perl;
use Text::CSV;
use MIME::Base64 qw(decode_base64);
use utf8;
use lib '.';
use ML::Tokeniser;
use Data::Dumper;

my $csv = Text::CSV->new();
open my $fhin, "<:encoding(utf8)", "email_summaries.csv" or die $!;
open my $fhout, ">:encoding(utf8)", "email_summary_ids.txt" or die $!;

my $model = "client_emails_16000.model";

my $tok = ML::Tokeniser->new(vocab_size => 276);
$tok->load(file => "$model");

my $sep_id = $tok->{special_tokens}{"[SEP]"};
die "no [SEP] defined in model" unless $sep_id;

my $processed = 0;
while (my $row = $csv->getline($fhin)) {
   my $body = decode_base64($row->[0]);
   utf8::decode($body);
   my $summary = decode_base64($row->[1]);
   utf8::decode($summary);
   print $fhout  join(",", @{$tok->encode(text => $body)}) . ",$sep_id," . join(",", @{$tok->encode(text => $summary)}) . "\n";
   $processed++;
}
close $fhout;

say "$processed processed";
