use Modern::Perl;
package ML::Util;
use Data::Dumper;
use Exporter 'import';
our @EXPORT_OK = qw(print_2d_array transpose add_2_arrays mult_2_arrays diagonal_matrix matmul 
                    print_1d_array rotate_matrix_180 conv2d linear add_constant div_constant 
                    adam_optimiser softmax scaled_dot_product_attention mult_constant randn
                    create_src_mask create_tgt_mask tokenize_sequence pad_sequence);

use Carp qw(croak confess);
use Math::Random::OO::Normal;
my $prng = Math::Random::OO::Normal->new();


my $eps = 1e-08;

sub adam_optimiser {
   my ($g, $m, $v, $x, $alpha, $beta1, $beta2, $t) = @_;
# g = gradient
# m & v = previous gradients
# x = weights
# beta1 & beta2 weighting of first and second order gradients to apply to new gradient
# alpha = learning rate
# eps = small fixed value, prevent / 0 error
# t = epoch number
# typical values: beta1 0.9, beta2 0.999 learning rate 0.1
   foreach my $i (0 .. $#$x) {
      foreach my $j (0 .. $#{$x->[0]}) {
confess "beta1 undefined" unless defined($beta1);
confess "beta2 undefined" unless defined($beta2);
confess "m $i $j undefined" unless defined($m->[$i][$j]);
confess "g $i $j undefined" . print_2d_array("g", $g) . print_2d_array("x", $x) unless defined($g->[$i][$j]);
confess "v $i $j undefined" unless defined($v->[$i][$j]);
         $m->[$i][$j] = $beta1 * $m->[$i][$j] + ( 1 - $beta1) * $g->[$i][$j];
         $v->[$i][$j] = $beta2 * $v->[$i][$j] + ( 1 - $beta2) * ($g->[$i][$j])**2;
         my $m_hat = $m->[$i][$j] / ( 1 - $beta1**$t);
         my $v_hat = $v->[$i][$j] / ( 1 - $beta2**$t);
         $x->[$i][$j] -= $alpha * $m_hat / (sqrt($v_hat) + $eps);
      }
   }
}

sub print_2d_array {
   my ($title,$d) = @_;
   say $title;
   if (ref($d->[0]) ne "ARRAY") {
      say "not a 2d array, ref(d) = " . ref($d) . " ref(d->[0]) = " . ref($d->[0]);
      say Dumper($d);
      confess "not a 2 d array " . join(",",caller());
   }
if (ref($d->[0][0]) eq "ARRAY") {
   die "not a 2d array, at least 3d " . join(",",caller());
}
   say "rows = " . scalar(@$d) . " columns = " . scalar(@{$d->[0]});
   my $row_num = 0;
   foreach my $row (@$d) {
      $row_num++;
      foreach my $col_num (0 .. $#{$row}) {
         if (!defined($row->[$col_num])) {
            confess "print_2d_array title row $row_num col $col_num not defined";
         }
      }
      say join("\t", map { ref($_)?join(",",@$_):sprintf("%+.5f",$_) } @{$row});
   }
   say "";
}

sub print_1d_array {
   my ($title,$d) = @_;
   say $title . " " . scalar(@$d) . " entries";
   say join("\t", map { sprintf("%+.5f",$_) } @{$d});
}

sub transpose {
   my $M = shift;   
   my $O = []; 
   foreach my $i (0 .. $#$M) {
if (ref($M->[0]) ne "ARRAY") {
   confess "ML::Util transpose input array not 2d";
}
      foreach my $j (0 .. $#{$M->[0]}) {
         $O->[$j][$i] = $M->[$i][$j];
      }
   }
   return $O;   
}

sub add_2_arrays { # same size assumed, or that M1 is smaller than M2
   my ($M1, $M2) = @_;
   my $O = [];
   foreach my $i (0 .. $#$M1) {
      foreach my $j (0 .. $#{$M1->[0]}) {
         if (!defined($M1->[$i][$j])) {
            say join(",", caller());
            confess "add_2_arrays M1 not defined at [$i][$j]";
         }
         if (!defined($M2->[$i][$j])) {
            say join(",", caller());
            confess "add_2_arrays M2 not defined at [$i][$j]";
         }
         $O->[$i][$j] = $M1->[$i][$j] + $M2->[$i][$j];
      } 
   }
   return $O;
} 

sub mult_2_arrays { # same size assumed, or that M1 is smaller than M2
   my ($M1, $M2) = @_;
   my $O = [];
   foreach my $i (0 .. $#$M1) {
      foreach my $j (0 .. $#{$M1->[0]}) {
if (!defined($M1->[$i][$j])) {
   say join(",", caller());
   say "M1 not defined at [$i][$j]";
die;
}
if (!defined($M2->[$i][$j])) {
   say join(",", caller());
   say "M2 not defined at [$i][$j]";
die;
}
         $O->[$i][$j] = $M1->[$i][$j] * $M2->[$i][$j];
      }
   }
   return $O;
}

sub weighted_add_2_arrays { # same size assumed
   my ($M1, $w1, $M2, $w2, $p1, $p2) = @_;
# adds two arrays, but multiplies each one by a weighting factor before adding
# optionally raises the matrix by a power before multiplying a weighting factor
   my $O = [];
   $p1 ||= 1;
   $p2 ||= 1;
   foreach my $i (0 .. $#$M1) {
      foreach my $j (0 .. $#{$M1->[0]}) {
if (!defined($M1->[$i][$j])) {
   say join(",", caller());
   say "M1 not defined at [$i][$j]";
die;
}
if (!defined($M2->[$i][$j])) {
   say join(",", caller());
   say "M2 not defined at [$i][$j]";
die;
}
         $O->[$i][$j] = $M1->[$i][$j]**$p1 * $w1 + $M2->[$i][$j]**$p2 * $w2;
      }
   }
   return $O;
}

sub update_weight_adam { # same size assumed
   my ($weights, $learning_rate, $m_hat, $v_hat, $eps) = @_;
# updates the input $weights based on the adam optimiser function
   foreach my $i (0 .. $#$weights) {
      foreach my $j (0 .. $#{$weights->[0]}) {
         $weights->[$i][$j] -= $learning_rate * $m_hat->[$i][$j] / (sqrt($v_hat->[$i][$j]) + $eps);
      }
   }
}


sub weight_array {
   my ($M1, $w1) = @_;
   my $O = [];
   foreach my $i (0 .. $#$M1) {
      foreach my $j (0 .. $#{$M1->[0]}) {
         $O->[$i][$j] = $M1->[$i][$j] * $w1;
      }
   }
   return $O;
}

sub diagonal_matrix {
   my ($value, $max_index) = @_;
   my $O = [];
   foreach my $i (0 .. $max_index) {
      foreach my $j (0 .. $max_index) {
         if ($i == $j) {
            $O->[$i][$j] = $value;
         } else {
            $O->[$i][$j] = 0;  
         }
      }
   }
   return $O;
}

sub matmul {
   # matrix multiplication
   my ($il, $ol) = @_;
   my $ar = $#$il;
  if (ref($il->[0]) ne "ARRAY") {
      say "not a 2d array, ref(il) = " . ref($il) . " ref(il->[0]) = " . ref($il->[0]);
      say Dumper($il);
      die "not a 2 d array " . join(",",caller());
   }
   my $ac = $#{$il->[0]};
   my $br = $#$ol;
   my $bc = $#{$ol->[0]};
   do {
      say "matmul: ar = $ar, ac = $ac, br = $br, bc = $bc";
      confess "matmul: input layer output not the same length as input of output layer, called by " . join(",",caller())
   } unless $ac == $br;
   my $c = [];
   foreach my $i ( 0 .. $ar) {
      foreach my $j ( 0  .. $bc) {
         foreach my $k ( 0 .. $ac) {
            $c->[$i][$j] += $il->[$i][$k] * $ol->[$k][$j];
if (!defined( $il->[$i][$k] ) ) {
   confess "matmul il->[$i][$k] not defined";
}  
if (!defined( $ol->[$k][$j] ) ) {
   confess "matmul ol->[$k][$j] not defined";
}  

#            say "c->[$i][$j] = " . $c->[$i][$j];
         }
      }
   }
   #say "matmul result = " . Dumper($c);
   return $c;
}

=pod
sub linear {
   # linear = matrix multiplication + addition: r = a x b + c
   my ($il, $ol, $c) = @_;
   my $ar = $#$il;
   my $ac = $#{$il->[0]};
   my $br = $#$ol;
   my $bc = $#{$ol->[0]};
   my $cr = $#{$c};
#   say "linear: ar = $ar, ac = $ac, br = $br, bc = $bc";
   do {
      say "linear: ar = $ar, ac = $ac, br = $br, bc = $bc";
      die "linear: input layer output not the same length as input of output layer, called by " . join(",",caller())
   } unless $ac == $br;
   do { 
      say "linear: ar = $ar, cr = $cr";
      die "linear: output layer rows not the same length as addition matrix, called by " , join(",", caller());
   } unless $ar == $cr;
   my $o = [];
   foreach my $i ( 0 .. $ar) {
      foreach my $j ( 0  .. $bc) {
         foreach my $k ( 0 .. $ac) {
            $o->[$i][$j] += $il->[$i][$k] * $ol->[$k][$j] + $c->[$i][0];
#            say "o->[$i][$j] = " . $o->[$i][$j];
         }
      }
   }
   #say "linear result = " . Dumper($c);
   return $o;
}
=cut

sub linear {
   # linear = matrix multiplication + addition: r = a x b + c
   my ($il, $ol, $c) = @_;
   my $ar = $#$il;
   my $ac = $#{$il->[0]};
   my $br = $#$ol;
   my $bc = $#{$ol->[0]};
   my $cr = $#{$c};
   #say "linear: ar = $ar, ac = $ac, br = $br, bc = $bc";
   do {
      say "linear: ar = $ar, ac = $ac, br = $br, bc = $bc";
      confess "linear: input 1 layer columns not the same length as rows of input layer 2, called by " . join(",",caller())
   } unless $ac == $br;
   do {
      say "linear: bc = $bc, cr = $cr";
      die "linear: output layer cols not the same length as addition matrix, called by " , join(",", caller());
   } unless $bc == $cr;
   my $o = [];
   foreach my $i ( 0 .. $ar) {
      foreach my $j ( 0  .. $bc) {
         $o->[$i][$j] = $c->[$j][0];
         foreach my $k ( 0 .. $ac) {
die "c->[$j][0] uninitialised " . join(", ", caller()) if !defined($c->[$j][0]);
            $o->[$i][$j] += $il->[$i][$k] * $ol->[$k][$j];
#            say "o->[$i][$j] = " . $o->[$i][$j];
         }
      }
   }
   #say "linear result = " . Dumper($c);
   return $o;
}

sub add_constant {
   my $A = shift;
   my $b = shift;
   my $A_x = $#{$A};
   my $A_y = $#{$A->[0]};
   foreach my $i (0 .. $A_x) {
      foreach my $j (0 .. $A_y) {
         $A->[$i][$j] += $b;
      }
   }
}

sub div_constant {
   my $A = shift;
   my $b = shift;
   my $A_x = $#{$A};
   my $A_y = $#{$A->[0]};
   foreach my $i (0 .. $A_x) {
      foreach my $j (0 .. $A_y) {
         $A->[$i][$j] /= $b;
      }
   }
}

sub mult_constant {
   my $A = shift;
   my $b = shift;
   my $A_x = $#{$A};
   my $A_y = $#{$A->[0]};
   foreach my $i (0 .. $A_x) {
      foreach my $j (0 .. $A_y) {
         $A->[$i][$j] *= $b;
      }
   }
}

sub rotate_matrix_180 {
   my $A = shift;
   my $A_rot = [];

   my $A_x = $#{$A};
   my $A_y = $#{$A->[0]};
   foreach my $i (0 .. $A_x) {
      foreach my $j (0 .. $A_y) {
         $A_rot->[$A_x - $i][$A_y - $j] = $A->[$i][$j];
      }
   }
   return $A_rot;
}

sub conv2d {
  my ($in, $input_size, $output_size, $stride, $filter, $operation) = @_;
  $operation = "padded" unless $operation eq "expand" or $operation eq "reduce";
  # operation:
  #    expand: the convolution starts at the $in(0, 0) cell, which means the output is $filter/2 cells bigger than $in in each axis
  #    padded: the convolution starts at the $in($filter/2, $filter/2) cell, so that $output is the same size as $in
  #    reduce: the convolution starts at the $in($filter, $filter) cell, so that the $output is $filter/2 cells smaller than $in in each axis
  # in each case the kernel centre is set to the cell to be calculated, so for $in(i, j) the cells involved in the calc are $in(i - filter/2, j - filter/2) to $in(i + filter/2, j + filter/2)
  # and any negative indicies are return "0".
  # this function assumes that the $filter has NOT been rotated, so the first action is to rotate the filter
  # without rotating the filter it would be "correlation" rather than "convolution"
  my $rot_filter = rotate_matrix_180($filter);
  my $kernel_size = scalar(@$filter);
  my $kernel_offset = int( $kernel_size  / 2); 
  my $output = [];
  my $in_offset = 0;
  if ($operation eq "expand") {
     $in_offset = $kernel_offset * 2;
  }
  if ($operation eq "padded") {
     $in_offset = $kernel_offset;
  }
  foreach my $i (0 .. $output_size - 1) {
     foreach my $j (0 .. $output_size - 1) {
        $output->[$i][$j] = 0;
        foreach my $u (0 .. $kernel_size - 1) {
           foreach my $v (0 .. $kernel_size - 1) {
              my $in_i = ($i * $stride) + $u - $in_offset;
              my $in_j = ($j * $stride) + $v - $in_offset;
              next if $in_i < 0 or $in_j < 0 or $in_i >= $input_size or $in_j >= $input_size;
         #     say "output->[$i][$j] += rot_filter->[$u][$v] " . $rot_filter->[$u][$v] . " * in->[$in_i][$in_j] " . $in->[$in_i][$in_j];
if (!defined($rot_filter->[$u][$v])) {
   die "rot_filter $u $v undefined - called by " . join(",",caller());
}
if (!defined($in->[$in_i][$in_j])) {
   die "in $in_i $in_j undefined - called by " . join(",",caller());
}
              $output->[$i][$j] += $rot_filter->[$u][$v] * $in->[$in_i][$in_j];
           }
        }
     }
  }
  return $output;
}
         
sub _softmax_by_row {
   my $input = shift;
   my $insize = scalar(@{$input});
   my $mini_batch_size = scalar(@{$input->[0]});
   my $max = [];
   my $sm_output = [];
   foreach my $i (0 .. $insize - 1) {
      $max->[$i] = $input->[$i][0];
      foreach my $j (0 .. $mini_batch_size - 1) {
         if ($input->[$i][$j] > $max->[$i]) {
            $max->[$i] = $input->[$i][$j];
         }
      }
   }
   my $sums = [];
   foreach my $i (0 .. $insize - 1) {
      $sums->[$i] = 0;
      foreach my $j (0 .. $mini_batch_size - 1) {
         $sums->[$i] += exp( $input->[$i][$j] - $max->[$i] );
      }
   }
   foreach my $i (0 .. $insize - 1) {
      foreach my $j (0 .. $mini_batch_size - 1) {
         $sm_output->[$i][$j] = exp( $input->[$i][$j] - $max->[$i]) / $sums->[$i];
      }
   }
   return $sm_output;
}

sub softmax {
   my $input = shift;
   my $by_row = shift;
   $by_row ||= 0;
   # defaults to generating softmax by column, by_row = 1 generates softmax by row
   if ($by_row) {
      return _softmax_by_row($input);
   }
   my $insize = scalar(@{$input});
   my $mini_batch_size = scalar(@{$input->[0]});
   my $max = [];
   my $sm_output = [];
   foreach my $i (0 .. $mini_batch_size - 1) {
      $max->[$i] = $input->[0][$i];
      foreach my $j (0 .. $insize - 1) {
         if ($input->[$j][$i] > $max->[$i]) {
            $max->[$i] = $input->[$j][$i];
         }
      }
   }
   my $sums = [];
   foreach my $i (0 .. $mini_batch_size - 1) {
      $sums->[$i] = 0;
      foreach my $j (0 .. $insize - 1) {
         $sums->[$i] += exp( $input->[$j][$i] - $max->[$i] );
      }
   }
   foreach my $i (0 .. $mini_batch_size - 1) {
      foreach my $j (0 .. $insize - 1) {
         $sm_output->[$j][$i] = exp( $input->[$j][$i] - $max->[$i]) / $sums->[$i];
      }
   }
   return $sm_output;
}

sub scaled_dot_product_attention {
   my ($query, $key, $value, $mask, $dropout) = @_;

   my $d_k = scalar(@{$query->[0][0][0]});
   my $attention_scores = [];
   foreach my $i (0 .. $#$query) {
      foreach my $j (0 .. $#{$query->[0]}) {
#print_2d_array("query $i $j", $query->[$i][$j]);
#print_2d_array("key $i $j", $key->[$i][$j]);
         $attention_scores->[$i][$j] = matmul( $query->[$i][$j], transpose($key->[$i][$j]));
#print_2d_array("attention scores part 1: $i $j", $attention_scores->[$i][$j]);
      }
   }
=pod
   say "attention_scores shape = " . join(", ", scalar(@{$attention_scores}),
                                                scalar(@{$attention_scores->[0]}),
                                                scalar(@{$attention_scores->[0][0]}),
                                                scalar(@{$attention_scores->[0][0][0]}));
=cut
   foreach my $i (0 .. $#$query) {
      foreach my $j (0 .. $#{$query->[0]}) {
         div_constant($attention_scores->[$i][$j], sqrt($d_k));
#print_2d_array("attention scores part 2: $i $j", $attention_scores->[$i][$j]);
      }
   }
   if (defined($mask)) {
      my @dims;
      my $d = $mask;
      while (ref($d) eq "ARRAY") {
         push @dims, scalar(@$d);
         $d = $d->[0];
      }
   #   say "mask shape:" . join(", ", @dims);
      if (scalar(@dims) == 2) { # 2-d mask - one row per batch
         foreach my $i (0 .. $#$query) {
            foreach my $j (0 .. $#{$query->[0]}) {
               foreach my $x (0 .. $#{$attention_scores->[0][0]}) {
                  foreach my $y (0 .. $#{$attention_scores->[0][0][0]}) {
                     $attention_scores->[$i][$j][$x][$y] -= $mask->[$i][$y] * 1e9;
                  }
               }
               #print_2d_array("attention scores part 3: $i $j", $attention_scores->[$i][$j]);
            }
         }
      } else { # 3-d mask x, y per batch
         foreach my $i (0 .. $#$query) {
            foreach my $j (0 .. $#{$query->[0]}) {
               foreach my $x (0 .. $#{$attention_scores->[0][0]}) {
                  foreach my $y (0 .. $#{$attention_scores->[0][0][0]}) {
                     # mask is 2d per batch entry
                     $attention_scores->[$i][$j][$x][$y] -= $mask->[$i][$x][$y] * 1e9;
                  }
               }
               #print_2d_array("attention scores part 3: $i $j", $attention_scores->[$i][$j]);
            }
         }
      }
   }
   my $attention_weights = [];
   foreach my $i (0 .. $#$query) {
      foreach my $j (0 .. $#{$query->[0]}) {
         #print_2d_array("attention_scores $i $j", $attention_scores->[$i][$j]);
         $attention_weights->[$i][$j] = softmax($attention_scores->[$i][$j], 1); # softmax by row
         #print_2d_array("attention_weights after softmax $i $j", $attention_weights->[$i][$j]);
      }
   }

   if (defined($dropout)) {
      foreach my $i (0 .. $#$query) {
         foreach my $j (0 .. $#{$query->[0]}) {
            foreach my $x (0 .. $#{$attention_weights->[0][0]}) {
               foreach my $y (0 .. $#{$attention_weights->[0][0][0]}) {
                  my $rand = rand();
                  if ($rand < $dropout) {
                     $attention_weights->[$i][$j][$x][$y] = 0;
                  }
               }
            }
            #print_2d_array("attention_weights after drop out $i $j", $attention_weights->[$i][$j]);
         }
      }
   }
   my $output = [];
   foreach my $i (0 .. $#$query) {
      foreach my $j (0 .. $#{$query->[0]}) {
         $output->[$i][$j] = matmul($attention_weights->[$i][$j], $value->[$i][$j]);
#print_2d_array("self attention output $i $j", $output->[$i][$j]);
      }
   }

   return $output, $attention_weights;

}

sub randn {
   my ($x, $y) = @_;
   my $O = [];
   foreach my $i (0 .. $x - 1) {
      foreach my $j (0 .. $y - 1) {
         $O->[$i][$j] = $prng->next;
      }
   }
   return $O;
}

sub create_src_mask {
   my %args = @_;
   # src_ids, pad_id
   my $mask = [];
   foreach my $b (0 .. scalar(@{$args{src_ids}}) - 1) { # batch size
      foreach my $s (0 .. scalar(@{$args{src_ids}->[0]}) - 1) { # seq len
         $mask->[$b][$s] = $args{src_ids}->[$b][$s] == $args{pad_id}?1:0;
      }
   }
   #my @mask = map { $_ == $args{pad_id}?1:0 } @{$args{src_ids}};
   return $mask; 
}

sub create_tgt_mask {
   my %args = @_; 
   #tgt_ids , pad_id
   my $batch_size = scalar(@{$args{tgt_ids}});
   my $tgt_seq_len = scalar(@{$args{tgt_ids}->[0]});
   my $tgt_mask = [];
   foreach my $b (0 .. $batch_size - 1) {
      #say "tgt_ids $b " . join(", " , @{$args{tgt_ids}->[$b]});
      my $tgt_padding_mask = [ map { $_ == $args{pad_id}?1:0 } @{$args{tgt_ids}->[$b]} ];
      foreach my $i (0 .. $tgt_seq_len - 1) {
         foreach my $j (0 .. $tgt_seq_len - 1) {
            $tgt_mask->[$b][$i][$j] = ($tgt_padding_mask->[$j] or $j>$i)?1:0;
         }
      }
      #print_2d_array("tgt mask $b", $tgt_mask->[$b]);
   }
   return $tgt_mask;
}

sub tokenize_sequence {
   my %args = @_;
# """Converts a sequence of tokens to a sequence of IDs."""
    return [ map { $args{token_to_id_map}->{$_} } @{$args{sequence}} ];
}

sub pad_sequence {
   my %args = @_;
#    """Pads a sequence of IDs to max_len.""" 
   my $padded_ids = [ @{$args{sequence_ids}},  ($args{pad_id}) x ($args{max_len} - scalar(@{$args{sequence_ids}})) ];
   return [ splice @$padded_ids, 0, $args{max_len} ]; 
}

1;
