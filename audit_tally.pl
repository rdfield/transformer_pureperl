#!/usr/bin/perl
# Mode-A spot-audit tally helper.
#
# Reads a decode_gen.pl output file (=== example N === / GEN: / REF: blocks),
# shows each GEN vs REF pair, and captures a single-keypress verdict:
#   f = faithful    a = acceptable    w = wrong    s = skip (unscored)
#   b = back (re-score previous)   q = quit & show tally   ? = help
# For a 'w', a follow-up key records WHY, so you get the split that decides
# whether the post-filter is enough:
#   l = literal fabrication (email/phone/PO/SKU invented -> post-filter catches)
#   m = mangled real entity (name/number garbled        -> needs more data)
#   o = other (missing key fact, degenerate, etc.)
#
# Verdicts are written to <auditfile>.results.csv after every keypress, so a
# crash or quit never loses work and re-running resumes where you left off.
#
# Usage: perl audit_tally.pl [spot_audit_100.txt]

use strict;
use warnings;

my $audit_file = shift @ARGV // '/home/mvine/embedding/spot_audit_100.txt';
my $results_file = "$audit_file.results.csv";

die "cannot read $audit_file\n" unless -r $audit_file;
die "need an interactive terminal for keypresses (don't pipe into me)\n"
    unless -t STDIN;

# ---- parse the decode file into ordered records -------------------------
open my $fh, '<', $audit_file or die "open $audit_file: $!\n";
my (@rec, $cur, $mode);
while (my $line = <$fh>) {
    if ($line =~ /^=== example\s+(\d+)\b.*?===/) {
        push @rec, $cur if $cur;
        $cur = { num => $1, head => $line, gen => '', ref => '' };
        $mode = undef;
        $cur->{head} =~ s/\s+$//;
    }
    elsif ($line =~ /^GEN:\s?(.*)$/) { $mode = 'gen'; $cur->{gen} = $1 if $cur; }
    elsif ($line =~ /^REF:\s?(.*)$/) { $mode = 'ref'; $cur->{ref} = $1 if $cur; }
    elsif ($cur && $mode) { $cur->{$mode} .= "\n" . $line =~ s/\s+$//r; }
}
push @rec, $cur if $cur;
close $fh;
die "no '=== example N ===' blocks found in $audit_file\n" unless @rec;

# ---- load any prior results (resume) ------------------------------------
my %score;   # num => [label, reason]
if (-r $results_file) {
    open my $rf, '<', $results_file or die "open $results_file: $!\n";
    while (<$rf>) {
        chomp;
        next if /^example,/;                 # header
        my ($n, $lab, $why) = split /,/, $_, 3;
        $score{$n} = [$lab // '', $why // ''] if defined $n && length $n;
    }
    close $rf;
}

sub save_results {
    open my $wf, '>', $results_file or die "write $results_file: $!\n";
    print $wf "example,label,reason\n";
    for my $r (@rec) {
        my $s = $score{$r->{num}} or next;
        print $wf join(',', $r->{num}, $s->[0], $s->[1]), "\n";
    }
    close $wf;
}

# ---- terminal raw mode (single keypress, no Enter) ----------------------
my $orig_stty = `stty -g 2>/dev/null`;
chomp $orig_stty;
sub raw_on  { system("stty -icanon -echo min 1 time 0 2>/dev/null"); }
sub raw_off {
    if ($orig_stty) { system("stty $orig_stty 2>/dev/null"); }
    else            { system("stty icanon echo 2>/dev/null"); }
}
END { raw_off(); }
$SIG{INT} = $SIG{TERM} = sub { raw_off(); print "\n"; exit 1; };
raw_on();

sub getkey { my $c = getc(STDIN); return defined $c ? lc $c : 'q'; }

# ---- colours (only if stdout is a tty) ----------------------------------
my $tty = -t STDOUT;
sub c { my ($code, $s) = @_; $tty ? "\e[${code}m$s\e[0m" : $s; }
sub clear { print $tty ? "\e[2J\e[H" : "\n" x 3; }

my %LABELNAME = (f => 'FAITHFUL', a => 'ACCEPTABLE', w => 'WRONG', s => 'skip');
my %WHYNAME   = (l => 'literal-fabrication', m => 'mangled-entity', o => 'other');

sub help_screen {
    clear();
    print c('1', "Spot-audit tally — keys\n\n");
    print "  ", c('32','f'), "  faithful    — send as-is, all key facts right\n";
    print "  ", c('33','a'), "  acceptable  — gist right, minor fixable errors\n";
    print "  ", c('31','w'), "  wrong       — missing key fact / hallucinated / degenerate\n";
    print "  ", c('36','s'), "  skip        — leave unscored for now\n";
    print "  ", c('36','b'), "  back        — return to previous item to re-score\n";
    print "  ", c('36','q'), "  quit        — save and show the tally\n";
    print "  ", c('36','?'), "  this help\n\n";
    print "After 'w' you'll be asked WHY: ",
          c('36','l'),"=literal-fabrication  ",
          c('36','m'),"=mangled-entity  ",
          c('36','o'),"=other  (Enter=unspecified)\n\n";
    print "press any key to continue...";
    getkey();
}

# ---- main loop ----------------------------------------------------------
# start at first unscored item
my $i = 0;
$i++ while $i < @rec && $score{$rec[$i]{num}};
$i = 0 if $i >= @rec;   # all scored: land on first, user can review/quit

while (1) {
    my $r   = $rec[$i];
    my $num = $r->{num};
    my $done = grep { $score{$_->{num}} } @rec;

    clear();
    printf "%s   %s   %s\n",
        c('1', "[$num]"),
        c('90', sprintf("%d/%d scored", scalar $done, scalar @rec)),
        c('90', "(item ".($i+1)." of ".scalar(@rec).")");
    print c('90', $r->{head}), "\n\n";
    print c('1;36', "GEN: "), $r->{gen}, "\n\n";
    print c('1;32', "REF: "), $r->{ref}, "\n\n";

    if (my $s = $score{$num}) {
        my $tag = $LABELNAME{$s->[0]} // $s->[0];
        $tag .= " ($WHYNAME{$s->[1]})" if $s->[1] && $WHYNAME{$s->[1]};
        print c('90', "current: $tag\n");
    }
    print c('90', "\n[f]aithful [a]cceptable [w]rong  [s]kip [b]ack [q]uit [?]help > ");

    my $k = getkey();

    if    ($k eq '?') { help_screen(); next; }
    elsif ($k eq 'q') { last; }
    elsif ($k eq 'b') { $i-- if $i > 0; next; }
    elsif ($k eq 's') { $score{$num} = ['skip','']; save_results(); }
    elsif ($k eq 'f' || $k eq 'a') { $score{$num} = [$k,'']; save_results(); }
    elsif ($k eq 'w') {
        print c('1;31', $k), "  why? [l]iteral [m]angled [o]ther (Enter=unspec) > ";
        my $why = getkey();
        $why = '' unless $why =~ /^[lmo]$/;
        $score{$num} = ['w', $why];
        save_results();
    }
    else { next; }   # unknown key: redraw, don't advance

    # advance to next unscored (or next item if all remaining are scored)
    if ($i + 1 < @rec) { $i++; }
    else { last; }   # scored the last item
}

raw_off();

# ---- tally --------------------------------------------------------------
my %n = (f=>0, a=>0, w=>0, skip=>0);
my %why = (l=>0, m=>0, o=>0, unspec=>0);
for my $r (@rec) {
    my $s = $score{$r->{num}} or next;
    $n{$s->[0]}++;
    if ($s->[0] eq 'w') {
        my $w = $s->[1];
        $why{ $w =~ /^[lmo]$/ ? $w : 'unspec' }++;
    }
}
my $scored = $n{f} + $n{a} + $n{w};
my $total  = scalar @rec;
my $pc = sub { $scored ? sprintf('%.1f%%', 100 * $_[0] / $scored) : 'n/a' };

clear();
print c('1', "Spot-audit tally  ($audit_file)\n");
print c('90', "results saved to $results_file\n\n");
printf "  scored:      %d / %d   (skipped %d, unscored %d)\n\n",
    $scored, $total, $n{skip}, $total - $scored - $n{skip};
printf "  %-12s %3d   %s\n", 'faithful',   $n{f}, $pc->($n{f});
printf "  %-12s %3d   %s\n", 'acceptable', $n{a}, $pc->($n{a});
printf "  %-12s %3d   %s\n", 'wrong',      $n{w}, $pc->($n{w});
print  "  ", '-' x 30, "\n";
printf "  %-12s %3d   %s   %s\n", 'faith+accept', $n{f}+$n{a}, $pc->($n{f}+$n{a}),
    c('1', '<- ship signal');
if ($n{w}) {
    print "\n  wrong breakdown ($n{w}):\n";
    printf "    %-34s %3d\n", 'literal fabrication (post-filter)', $why{l};
    printf "    %-34s %3d\n", 'mangled entity (needs data)',       $why{m};
    printf "    %-34s %3d\n", 'other',                             $why{o};
    printf "    %-34s %3d\n", 'unspecified',                       $why{unspec} if $why{unspec};
}
print "\n";
