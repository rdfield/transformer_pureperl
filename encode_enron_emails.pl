use Modern::Perl;
use Text::CSV;
use lib '/home/mvine/embedding';
use ML::Tokeniser;
use Text::CSV;
use utf8;
use Digest::MD5 qw(md5 md5_hex md5_base64);


my $csv = Text::CSV->new({ binary => 1});

my $digests = {};

open my $fhin, "<:encoding(utf8)", "enron_emails.csv" or die $!;
my $processed = 0;
while (my $row = $csv->getline($fhin)) {
   $processed++;
   do { say "skipping headers"; next;} if $processed <= 1;
   my $txt_body = $row->[1];
   say "=========== original ====================";
   say "$txt_body\n"; 

# strip everything before the first blank line:
   $txt_body =~ s/(.*?^$)//sm;
#   say "=========== trimmed stage 1 ====================";
#   say "$txt_body\n"; 

# strip the "> " at the start of forwarded lines
   $txt_body =~ s/(^>\s+)//smg;
#   say "=========== trimmed stage 2 ====================";
#   say "$txt_body\n"; 
# strip any line that looks like a header, e.g. Subject: some text
   $txt_body =~ s/(^\s*?[\w\=]+:\s.*?$)//msg;
#   say "=========== trimmed stage 3 ====================";
#   say "$txt_body\n"; 
# strip any "Forwarded by" lines
   $txt_body =~ s/(^.*?Forwarded by.*?$)//imsg;
#   say "=========== trimmed stage 4 ====================";
#   say "$txt_body\n"; 
# strip old style "on <date> <name> wrote" type lines
   $txt_body =~ s/(^".*?M$)//imsg;
#   say "=========== trimmed stage 5 ====================";
#   say "$txt_body\n"; 
# strip lines with extended non-word chars, e.g. ----------
   $txt_body =~ s/(^.*?\p{P}{5,}.*?$)//imsg;
#   say "=========== trimmed stage 6 ====================";
#   say "$txt_body\n"; 

# final tweaks:

# lots of "mailto" sections at the top of emails
$txt_body =~ s/(^.*?\<mailto:.*?$)//ms;
# strip blank lines
   $txt_body =~ s/\n\n/\n/g;

   say "=========== final trimmed  ====================";
   say "$txt_body\n"; 
   say "=========================================\n";

die if $processed > 10000;
}
close $fhin;




