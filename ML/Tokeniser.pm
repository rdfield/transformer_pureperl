package ML::Tokeniser;
use Modern::Perl;
#use lib '..';
use parent qw(ML::BaseTokeniser);
use Data::Dumper;
use feature 'unicode_strings';
use Time::HiRes qw(time);
use Carp qw(carp);

# GPT-4's pre-tokenisation split, as used by minbpe. Kept as a string so that
# save() writes it into the .model file and load() restores it: encoding has to
# split text exactly the way training did, or merges land in the wrong places.
our $GPT4_SPLIT_PATTERN_STR = q{'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+};
our $GPT4_SPLIT_PATTERN = qr/$GPT4_SPLIT_PATTERN_STR/;

sub new {
   my $class = shift;
   my $self = $class->SUPER::new(@_);
   my %args = @_;
   $self->{pattern} = $args{pattern} // $GPT4_SPLIT_PATTERN_STR;
   $self->{special_tokens} = {};
   $self->{inverse_special_tokens} = {};
die unless $self->{vocab_size} =~ /^\d+$/;
   return $self;
}

sub merges {
   my $self = shift;
   return $self->{merges};
}

# load() replaces the merges, so the cached merge lookup must go with them, or a
# reused tokeniser keeps encoding with the previous model's merges. A pre-fix
# .model file has an empty pattern line; those were trained with GPT-4's.
sub load {
   my $self = shift;
   $self->SUPER::load(@_);
   delete $self->{merges_hash};
   $self->{pattern} = $GPT4_SPLIT_PATTERN_STR unless length($self->{pattern} // '');
}

sub train {
   my $self = shift;

  my $t_start = time;

   my %params = @_;
   my $text = $params{text};
   die "vocab_size must be >= 256" unless $self->{vocab_size} >= 256;
   my $num_merges = $self->{vocab_size} - 256;

  my $t = time;

   delete $self->{merges_hash};                 # train() replaces the merges
   $self->_check_decoded($text, 'train');
   my $split_re = $self->_split_re;
   my @chunks = $text =~ /$split_re/gms;

  $t = time;
   # Unique chunks in FIRST-SEEN order, not hash order: equally frequent pairs
   # are tie-broken by which was seen first, so hash order made the same corpus
   # train a different vocab on every run. First-seen is also minbpe's rule.
   my (%freq, @order, @weight, @ids);
   for (@chunks) { push @order, $_ unless $freq{$_}++ }
   foreach my $ch (@order) {
      push @ids , [ map { utf8::encode($_); unpack('C*', $_)} split//,$ch ];
      push @weight, $freq{$ch};
   } 
   if ($self->{verbose}) {
      foreach my $ch (@ids) {
         print "[" . join(", ",@$ch) . "], ";
      }
      print "\n";
   }
   $self->{merges} = [];
   my $stats = {};
   my $pair_to_chunks = {};
   $t = time;

  for my $ci (0 .. $#ids) {
      my $ch = $ids[$ci];
      for my $i (0 .. $#$ch - 1) {
          my $key = "$ch->[$i],$ch->[$i+1]";
          $stats->{$key}{count}+=$weight[$ci];
          #if ($stats->{$key}{count} == 1) {
             $stats->{$key}{mergeidx} //= $self->{mergeidx}++;
          #}
          $pair_to_chunks->{$key}{$ci} = 1;
      }
  }

   foreach my $i (0 .. $num_merges - 1) {
      $t = time;
      if ($i % 50 == 0) {
         print ".";
      }
      my $most_frequent_key;
      foreach my $key (keys %$stats) {
         if (!defined($most_frequent_key)) {
            $most_frequent_key = $key;
         } else {
            if ($stats->{$key}->{count} > $stats->{$most_frequent_key}->{count}
                or ( $stats->{$key}->{count} == $stats->{$most_frequent_key}->{count}
                     and $stats->{$key}->{mergeidx} < $stats->{$most_frequent_key}{mergeidx}) ) {
               $most_frequent_key = $key;
            }
         }
      }
      my $most_frequent_pair = [ split/,/, $most_frequent_key ];
     
      say "pair_to_find = " . join(", ", @$most_frequent_pair) if $self->{verbose};

      my $idx = 256 + $i;
      $t = time;
      # chunk order decides the first-seen order of the pairs this merge creates
      my @affected = sort { $a <=> $b } keys %{$pair_to_chunks->{$most_frequent_key}};
      if (scalar(@affected) > 0) {
         foreach my $ch (@affected) {
            $ids[$ch] = $self->merge($ids[$ch], $most_frequent_pair, $idx, $stats, $pair_to_chunks, $ch, $weight[$ch]);
         }
      }
      #push @{$self->{merges}}, { key => $sorted[0]->{key}, pair => $most_frequent_pair, value => $idx };
      push @{$self->{merges}}, { key => $most_frequent_key, pair => $most_frequent_pair, value => $idx };
      $self->{vocab}[$idx] = $self->{vocab}[$most_frequent_pair->[0]] . $self->{vocab}[$most_frequent_pair->[1]];
      if ($self->{verbose}) {
         say "merge " . ($i+1) . "/$num_merges: (" . join(", ", @$most_frequent_pair) .") -> $idx (" . $self->{vocab}[$idx] . ") had " . $stats->{$most_frequent_key}->{count} . " occurrences";
      }
   }
   if ($self->{verbose}) {
      foreach my $ch (@ids) {
         print "[" . join(", ",@$ch) . "], ";
      }
      print "\n";
   }
}

sub _build_merges_hash {
   my $self = shift;
   my %h;
   for my $merge (@{$self->{merges}}) {
       $h{ $merge->{key} } = $merge;
   }
   $self->{merges_hash} = \%h;
}

# Encode one ordinary (non-special) segment. Mirrors minbpe's
# RegexTokenizer.encode_ordinary: split with the same pattern train() used,
# then encode each piece on its own, so no merge crosses a piece boundary.
#
# Before 2026-10-23 this applied, at each step, the merge whose pair was MOST
# FREQUENT in the text being encoded -- the training rule -- and ran over the
# whole text unsplit. The result decoded correctly, so nothing caught it, but it
# was non-canonical: a word's ids depended on the rest of the text, it used
# about 1.4x more tokens than the vocab allows, and its cost grew with the
# square of the text length.
sub _encode {
   my ($self, $text) = @_;
   $self->_build_merges_hash unless exists $self->{merges_hash};
   my $split_re = $self->_split_re;
   return map { $self->_encode_chunk($_) } ($text =~ /$split_re/gms);
}

# minbpe's _encode_chunk: repeatedly apply the EARLIEST-learned merge present.
# How often a pair occurs in the text plays no part -- that is a training rule.
# Merge ids are assigned in training order, so the lowest id is the earliest.
sub _encode_chunk {
   my ($self, $chunk) = @_;
   utf8::encode($chunk);                 # characters -> UTF-8 bytes (a copy)
   my @ids = unpack('C*', $chunk);
   while (@ids >= 2) {
      my $best;
      for my $i (0 .. $#ids - 1) {
         my $m = $self->{merges_hash}{"$ids[$i],$ids[$i+1]"} or next;
         $best = $m if !$best || $m->{value} < $best->{value};
      }
      last unless $best;
      @ids = $self->merge_for_encode(\@ids, $best->{pair}, $best->{value});
   }
   return @ids;
}

# The compiled split pattern. load() restores the pattern from the .model
# file; files saved before the pattern was written there have an empty line,
# and were all trained with the GPT-4 pattern, so that is the fallback.
sub _split_re {
   my $self = shift;
   $self->{pattern} = $GPT4_SPLIT_PATTERN_STR unless length($self->{pattern} // '');
   return $self->{_split_re}{ $self->{pattern} } //= qr/$self->{pattern}/;
}

# train() and encode() take decoded (character) strings. A string that still
# holds raw UTF-8 bytes has every non-ASCII character encoded to UTF-8 a second
# time, which trains merges that can never match decoded text. Raw UTF-8 shows
# as a lead byte followed by a continuation byte, which decoded text almost
# never contains. Warns once per tokeniser.
sub _check_decoded {
   my ($self, $text, $who) = @_;
   return if $self->{_warned_undecoded} or utf8::is_utf8($text) or $text !~ /[\xC2-\xF4][\x80-\xBF]/;
   $self->{_warned_undecoded} = 1;
   carp "ML::Tokeniser::$who: the text looks like undecoded UTF-8 bytes. Decode it first "
      . "(e.g. read it with :encoding(UTF-8)), or every non-ASCII character is tokenised double-encoded";
}

sub encode {
   my ($self) = shift;
   my %params = @_;
   my ($text) = $params{text} or die;
   $self->_check_decoded($text, 'encode');
   my %options = @_;
   my $special = {};
   $options{allowed_special} ||= "none_raise";
   my $allowed_special = $options{allowed_special};
   if ($allowed_special eq "all") {
      $special = $self->{special_tokens};
   } elsif ( $allowed_special eq "none" ) {
   } elsif ( $allowed_special eq "none_raise" ) {
      foreach my $tv (keys %{$self->{special_tokens}}) {
         if (index($text, $tv) >= 0) {
            die "none_raise set, but special token $tv found in text at " . index($text, $tv);
         }
      }
   } elsif ( ref($allowed_special) eq "ARRAY" ) {
      foreach my $st (keys %{$self->{special_tokens}}) {
         if ( grep { $st eq $_ } @$allowed_special ) {
            $special->{ $st } = $self->{special_tokens}{$st};
         }
      }
   } else {
      die "unknown value for 'allowed_special'";
   }
   if (!scalar(%$special)) {
      my @result = $self->_encode($text);
      return wantarray ? @result : \@result;
   }
   my $split_text = "(" . join('|', map { quotemeta($_) } keys %$special ) . ')';
   my $split_re = qr{$split_text};
   my @chunks = split $split_re, $text;
   my @ids = ();
   foreach my $ch (@chunks) {
      if (grep {$ch eq $_} keys %$special) {
          push @ids, $special->{$ch};
      } else {
          push @ids, $self->_encode($ch);
      }
   }
   if (wantarray) {
      return @ids;
   } else {
      return \@ids;
   }
}

sub decode {
   my ($self) = shift;
   my %params = @_;
   my @ids = @{$params{ids}};
   my $tokens = "";
   foreach my $id (@ids) {
      if (grep{ $id eq $_ } keys %{$self->{inverse_special_tokens}}) {
         $tokens .= $self->{inverse_special_tokens}{$id};
      } else {
         $tokens .= $self->{vocab}[$id];
      }
   }
   utf8::decode($tokens);
   return $tokens;
}

sub register_special_tokens {
   my ($self, $special_tokens) = @_;
   # special_tokens is a dictionary of str -> int
   # example: {"<|endoftext|>": 100257}
   $self->{special_tokens} = $special_tokens;
   %{$self->{inverse_special_tokens}} = reverse %{$self->{special_tokens}};
}

1;
