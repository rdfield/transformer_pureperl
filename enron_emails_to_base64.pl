use Modern::Perl;
use lib '.';
use ML::Tokeniser;
use JSON::XS; 
use MIME::Base64 qw(encode_base64);

my $filename = 'cleaned_enron_emails.json';

open(my $fh, '<:encoding(UTF-8)', $filename) or die $!;
my $json_text = do { local $/; <$fh> };
close($fh);

# Decode the string
my $data = decode_json($json_text);

say "data decoded";
my $processed = 0;
open FH, ">", "enron_emails_base64.txt";
foreach my $e (@$data) {
   #say $e->{Body};
   utf8::encode($e->{Body}) if utf8::is_utf8($e->{Body});
   print FH encode_base64($e->{Body},'') . "\n";
}
close FH;
