use Modern::Perl;
use Text::CSV;
use MIME::Base64 qw(decode_base64);
use utf8;

my $csv = Text::CSV->new();
open my $fhin, "<:encoding(utf8)", "email_categories.csv" or die $!;
open my $fhout, ">:encoding(utf8)", "emails.txt" or die $!;
while (my $row = $csv->getline($fhin)) {
   my $body = decode_base64($row->[2]);
   utf8::decode($body);
   
   print $fhout  $row->[1] . "\n$body\n";
}
close $fhout;
