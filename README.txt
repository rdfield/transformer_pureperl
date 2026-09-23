Pre-Training
============

original data file: cleaned_enron_emails.json
contains metadata and body of all the Enron emails (517000)

convert email bodies to base64 (so one per line) using: enron_emails_to_base64.pl

which creates the file "enron_emails_base64.txt" - one email body per line, so it can be "sharded" and converted to token ids in parallel.

there are 2 token models: 
 - client_emails_16000.model
 - client_emails_8000.model
with 16000 and 8000 tokens respectively.  The 16000 was generated to try and help with the lack of CLIENT training emails by giving it more depth of data.

It looks like 8000 is the way to go, since 16000 didn't really help and uses all of the available VRAM - 12GB - for training on even a small model.

8000 uses -  GB - on the same batch size = 8, d=256, layers = 4 transformer model.

Using the model to create the token ids for each email:

convert_enron_base64_to_ids.sh  enron_emails_base64.txt 12

the second parameter is the number of shards (using a multiple of the number of cpu cores available helps)

De-duplicate the input data and split the remaining lines into training and validation datasets (defaults to 10% validation):

perl dedup_singles.pl enron_emails_base64.txt.encoded enron_full.train enron_full.val
(original 516794 emails dedup'd down to 199570)



