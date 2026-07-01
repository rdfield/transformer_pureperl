#!/usr/bin/perl
  # stratified_split.pl <input.csv> <train.csv> <val.csv> [val_fraction]
  #
  # Reads a tokenised CSV (one email per line, format: class_id,tok,tok,tok,...).
  # Stratified split: per-class shuffle, then split each class's emails by the
  # given fraction into train and val.

  use strict;
  use warnings;
  use List::Util qw(shuffle);

  my ($in_path, $train_path, $val_path, $val_fraction) = @ARGV;
  $val_fraction //= 0.20;
  die "usage: $0 <input.csv> <train.csv> <val.csv> [val_fraction]\n"
      unless $in_path && $train_path && $val_path;

  # Bucket by class
  my %by_class;
  open(my $in, '<', $in_path) or die "cannot read $in_path: $!";
  while (my $line = <$in>) {
      chomp $line;
      next if $line eq '';
      my ($class_id) = split /,/, $line, 2;
      push @{$by_class{$class_id}}, $line;
  }
  close $in;

  # Per-class stratified split
  my (@train_rows, @val_rows);
  print "Stratified split (val_fraction = $val_fraction):\n";
  for my $cls (sort { $a <=> $b } keys %by_class) {
      my @rows = shuffle @{$by_class{$cls}};
      my $n_val = int(scalar(@rows) * $val_fraction);
      # Always at least 1 val sample per class (unless class has 0)
      $n_val = 1 if $n_val == 0 && scalar(@rows) > 0;
      # Never put EVERY sample in val
      $n_val = scalar(@rows) - 1 if $n_val >= scalar(@rows) && scalar(@rows) > 1;

      my @val_part   = @rows[0 .. $n_val - 1];
      my @train_part = @rows[$n_val .. $#rows];

      printf "  class %2d: total=%5d  train=%5d  val=%5d\n",
          $cls, scalar(@rows), scalar(@train_part), scalar(@val_part);

      push @train_rows, @train_part;
      push @val_rows,   @val_part;
  }

  # Final shuffle of train (matters for SGD); val order doesn't matter
  @train_rows = shuffle @train_rows;

  open(my $tr, '>', $train_path) or die "cannot write $train_path: $!";
  print $tr join("\n", @train_rows), "\n";
  close $tr;

  open(my $v, '>', $val_path) or die "cannot write $val_path: $!";
  print $v join("\n", @val_rows), "\n";
  close $v;

  printf "Wrote %d train rows to %s\n", scalar(@train_rows), $train_path;
  printf "Wrote %d val rows to %s\n",   scalar(@val_rows),   $val_path;

