#!/usr/bin/perl
# Stage 1 of the teacher bake-off: generate a candidate summary for every thread
# in the sample, via an OpenAI-compatible endpoint (OpenRouter by default).
#
# THE PROMPT COMES FROM A FILE YOU MAINTAIN, AND ITS HASH IS RECORDED.
# The authoritative prompt lives on the production server, which is behind jump
# boxes and unreachable from here — and the local copy in
# truncate_emails_regen_summary.pl is KNOWN STALE (it still carries "in two
# sentences or fewer", removed in prod after customer feedback). So the prompt is
# supplied as a plain text file and a SHA of it is written into every output row:
# a candidate judged on a different prompt is a test of prompt+model rather than
# of the model, and the hash is what lets you prove afterwards that all arms
# shared one. Keep bakeoff_prompt.txt in step with prod by hand.
#
# --from-script extracts from the Perl script instead, with a warning. Use it only
# if you have just confirmed that script matches prod.
#
# THE API KEY COMES FROM THE ENVIRONMENT AND IS NEVER WRITTEN ANYWHERE:
#   export OPENROUTER_API_KEY=...
# (The production script still has its key inline on line 36. That is a separate
# problem — it is gitignored, and the key wants rotating since it has been sitting
# on disk in the clear. Do not copy that pattern here.)
#
# Resumable: on restart, ids already present in the output are skipped, so a run
# interrupted by a rate limit continues rather than starting over. That matters
# most on free models, where the per-minute cap is the binding constraint.
#
# temperature 0 for reproducibility. Latency and token counts are recorded per
# item because they are the throughput and cost numbers for extrapolating a full
# 50k run, and they come free here.
#
# Output CSV: id,model,summary_b64,ms,prompt_tokens,completion_tokens,finish,prompt_sha
#
#   export OPENROUTER_API_KEY=...
#   perl bakeoff_generate.pl sample.csv out.csv --model qwen/qwen3-235b-a22b-2507
#        [--prompt-file bakeoff_prompt.txt | --from-script F]
#        [--base-url URL] [--max-tokens 400] [--sleep 0.2] [--retries 5] [--limit N]
#
use Modern::Perl;
use Getopt::Long;
use Text::CSV;
use MIME::Base64 qw(encode_base64 decode_base64);
use LWP::UserAgent;
use HTTP::Request;
use JSON::PP qw(encode_json decode_json);
use Time::HiRes qw(time sleep);
use Digest::SHA qw(sha1_hex);

my $MODEL       = '';
my $BASE        = 'https://openrouter.ai/api/v1/chat/completions';
my $PROMPT_FILE = 'bakeoff_prompt.txt';
my $FROM_SCRIPT = '';
my $MAX_TOK     = 400;
my $SLEEP       = 0.2;
my $RETRIES     = 5;
my $LIMIT       = 0;
my $TIMEOUT     = 180;
GetOptions('model=s' => \$MODEL, 'base-url=s' => \$BASE, 'prompt-file=s' => \$PROMPT_FILE,
           'from-script=s' => \$FROM_SCRIPT,
           'max-tokens=i' => \$MAX_TOK, 'sleep=f' => \$SLEEP, 'retries=i' => \$RETRIES,
           'limit=i' => \$LIMIT, 'timeout=i' => \$TIMEOUT) or die usage();

my $in  = shift @ARGV or die usage();
my $out = shift @ARGV or die usage();
die "--model is required\n" unless $MODEL;
die "input $in not found\n" unless -e $in;
die "refusing to overwrite input\n" if $out eq $in;

my $KEY = $ENV{OPENROUTER_API_KEY}
   or die "OPENROUTER_API_KEY is not set. export it; do not hardcode it.\n";

my ($prompt, $prompt_src);
if ($FROM_SCRIPT) {
   $prompt = extract_prompt($FROM_SCRIPT);
   $prompt_src = $FROM_SCRIPT;
   say "WARNING: extracting the prompt from $FROM_SCRIPT. The local copy has been";
   say "         stale before now — confirm it matches production before trusting";
   say "         any result from this run.";
} else {
   die "prompt file $PROMPT_FILE not found (write the CURRENT production prompt to it)\n"
      unless -e $PROMPT_FILE;
   open my $pf, '<:encoding(UTF-8)', $PROMPT_FILE or die "$PROMPT_FILE: $!";
   local $/; $prompt = <$pf>; close $pf;
   $prompt =~ s/\s+\z//;
   $prompt_src = $PROMPT_FILE;
}
die "prompt is empty\n" unless length $prompt;
my $pbytes = $prompt; utf8::encode($pbytes);
my $PSHA = substr(sha1_hex($pbytes), 0, 12);
say "prompt: " . length($prompt) . " chars, sha $PSHA, from $prompt_src";
say "model:  $MODEL";

# ---- resume: which ids are already done? ----
my %done;
if (-e $out) {
   my $rcsv = Text::CSV->new({ binary => 1 });
   open my $rfh, '<', $out or die "$out: $!";
   while (my $r = $rcsv->getline($rfh)) { $done{ $r->[0] } = 1 if @$r >= 3 }
   close $rfh;
   say "resuming: " . scalar(keys %done) . " already done";
}

my $csv  = Text::CSV->new({ binary => 1 });
my $ocsv = Text::CSV->new({ binary => 1, eol => "\n" });
open my $fh,  '<', $in or die "$in: $!";
open my $ofh, '>>', $out or die "$out: $!";
$ofh->autoflush(1);          # a rate-limit kill must not lose completed work

my $ua = LWP::UserAgent->new(timeout => $TIMEOUT);
my ($n, $ok, $fail, $skip, $ptok, $ctok, $t_ms) = (0) x 7;
my $t0 = time;

while (my $r = $csv->getline($fh)) {
   next unless @$r >= 4;
   my ($id, $band, $body_b64) = @$r[0, 1, 2];
   if ($done{$id}) { $skip++; next }
   last if $LIMIT && $n >= $LIMIT;
   $n++;

   my $body = decode_base64($body_b64);
   utf8::decode($body);

   # Mirror the production call exactly: the prompt and the email text are one
   # single user turn, prompt first, separated by a space.
   # reasoning OFF. Several Qwen models reason by default and will spend the
   # ENTIRE max_tokens budget on a `reasoning` field, returning content: null and
   # finish_reason: length. Measured on qwen3.7-flash: 200 output tokens of
   # thinking and an empty summary, against 21 tokens and a correct summary with
   # reasoning disabled — 7x the cost for nothing. Harmless on models that do not
   # reason (the 235B ignores it).
   my $payload = encode_json({
      model       => $MODEL,
      temperature => 0,
      max_tokens  => $MAX_TOK,
      reasoning   => { enabled => \0 },
      messages    => [ { role => 'user', content => $prompt . ' ' . $body } ],
   });

   my ($summary, $fin, $pt, $ct, $ms) = call_with_retry($payload);
   if (defined $summary) {
      $ok++; $ptok += $pt; $ctok += $ct; $t_ms += $ms;
      my $s = $summary; utf8::encode($s);
      $ocsv->print($ofh, [ $id, $MODEL, encode_base64($s, ''), int($ms), $pt, $ct, $fin, $PSHA ]);
   } else {
      $fail++;
      say "  id $id: FAILED after $RETRIES retries";
   }
   printf "  %3d/%s  id %-8s band %-10s %5dms  %s\n",
          $n, ($LIMIT || '?'), $id, $band, $ms,
          defined $summary ? sprintf('%d chars', length $summary) : 'FAIL';
   sleep($SLEEP) if $SLEEP;
}
close $fh; close $ofh;

my $wall = time - $t0;
say "";
say sprintf("done: %d ok, %d failed, %d skipped (already present)", $ok, $fail, $skip);
say sprintf("tokens: %d prompt, %d completion", $ptok, $ctok);
say sprintf("latency: mean %.0f ms/item, wall %.0f s", $ok ? $t_ms / $ok : 0, $wall);
if ($ok) {
   # extrapolation to a full run, which is the decision this feeds
   my $per = ($ptok + $ctok) / $ok;
   say sprintf("extrapolated to 50,000 items: %.1fM prompt + %.1fM completion tokens, %.1f h wall",
               $ptok / $ok * 50_000 / 1e6, $ctok / $ok * 50_000 / 1e6,
               $wall / $ok * 50_000 / 3600);
   say "  (wall is single-stream; concurrency would cut it, cost would not change)";
}
say "wrote $out";

# ---------------------------------------------------------------------------
# Pull `my $prompt = "...";` out of the production script. Deliberately narrow:
# it must match one single-line double-quoted assignment, and it must not be the
# line holding the api_key.
sub extract_prompt {
   my $file = shift;
   die "prompt file $file not found\n" unless -e $file;
   open my $pfh, '<', $file or die "$file: $!";
   my $found;
   while (my $l = <$pfh>) {
      next if $l =~ /api_key/;
      if ($l =~ /^\s*my\s+\$prompt\s*=\s*"(.+)"\s*;\s*$/) {
         $found = $1;
         last;
      }
   }
   close $pfh;
   die "could not find a single-line `my \$prompt = \"...\";` in $file\n" unless defined $found;
   $found =~ s/\\"/"/g;
   $found =~ s/\\\\/\\/g;
   return $found;
}

sub call_with_retry {
   my $payload = shift;
   my $wait = 2;
   for my $attempt (1 .. $RETRIES) {
      my $req = HTTP::Request->new(POST => $BASE);
      $req->header('Authorization' => "Bearer $KEY");
      $req->content_type('application/json');
      $req->content($payload);
      my $t = time;
      my $res = $ua->request($req);
      my $ms = (time - $t) * 1000;

      if ($res->is_success) {
         my $d = eval { decode_json($res->decoded_content) };
         unless ($d) { say "  bad JSON, retrying"; sleep($wait); $wait *= 2; next }
         if (my $e = $d->{error}{message}) {
            say "  api error: $e";
            sleep($wait); $wait *= 2; next;
         }
         my $text = $d->{choices}[0]{message}{content};
         unless (defined $text && length $text) {
            my $fin = $d->{choices}[0]{finish_reason} // '?';
            my $rsn = $d->{choices}[0]{message}{reasoning};
            say $rsn ? "  EMPTY content but reasoning present (finish=$fin) — the model"
                     . " reasoned away its whole budget; reasoning should be disabled"
                     : "  empty completion (finish=$fin), retrying";
            sleep($wait); $wait *= 2; next;
         }
         $text =~ s/^\s+|\s+$//g;
         return ($text, $d->{choices}[0]{finish_reason} // '',
                 $d->{usage}{prompt_tokens} // 0, $d->{usage}{completion_tokens} // 0, $ms);
      }

      my $code = $res->code;
      # 429 and 5xx are worth waiting out; anything else is our fault, so stop.
      if ($code == 429 || $code >= 500) {
         my $ra = $res->header('Retry-After');
         my $w = ($ra && $ra =~ /^\d+$/) ? $ra : $wait;
         say "  HTTP $code, waiting ${w}s (attempt $attempt/$RETRIES)";
         sleep($w); $wait *= 2; next;
      }
      say "  HTTP $code: " . substr($res->decoded_content // '', 0, 200);
      return (undef, '', 0, 0, $ms);
   }
   return (undef, '', 0, 0, 0);
}

sub usage {
   return "usage: OPENROUTER_API_KEY=... $0 sample.csv out.csv --model ID\n" .
          "       [--prompt-file bakeoff_prompt.txt | --from-script F]\n" .
          "       [--base-url URL] [--max-tokens 400]\n" .
          "       [--sleep 0.2] [--retries 5] [--limit N] [--timeout 180]\n";
}
