package ML::MHA;
use Modern::Perl;
use ML::Linear;
use ML::Util qw(print_2d_array scaled_dot_product_attention add_2_arrays randn matmul transpose mult_2_arrays adam_optimiser);
use Storable qw(dclone);
use Carp qw(confess);

my $input_limit = 1500;

# optimise the following:  w_Q, w_K, w_V and w_O

sub get_weights {
   my $self = shift;
   return { 
      max_seq_length => $self->{insize},
      embeddings => $self->{embeddings},
      num_heads => $self->{num_heads},
      dropout => $self->{dropout},
      W_q => $self->{W_q},
      W_v => $self->{W_v},
      W_k => $self->{W_k},
      W_o => $self->{W_o}
   }
}

sub new {
   my $class = shift;
   my $self = {};
   my %args = @_;
   $self->{name} = $args{name} || "MHA";
   my $debug = $args{debug};
   if (defined($debug) and $debug =~ /^\d+$/ and $debug > 0) {
      $debug = 1;
   } else {
      $debug = 0;
   }
   $self->{debug} = $debug;
   my $input_size = $args{max_seq_length};
   if (!defined($input_size) or $input_size !~ /^\d+$/ or $input_size > $input_limit) {
      die "Invalid embeddings parameter '$input_size', must be integer <= $input_limit";
   }
   $self->{insize} = $input_size; # max sequence length, for reserving enough memory
   $self->{created_by}  = join(", ", caller());
   my $embeddings = $args{embeddings};
   if (!defined($embeddings) or $embeddings !~ /^\d+$/) {
      die "Invalid embeddings parameter '$embeddings', must be integer ";
   }
   $self->{embeddings} = $embeddings;
   my $num_heads = $args{num_heads};
   if (!defined($num_heads) or $num_heads !~ /^\d+$/) {
      confess "Invalid num_heads parameter '$num_heads', must be integer ";
   }
   $self->{num_heads} = $num_heads;

   my $dropout = $args{dropout};
   if (!defined($dropout) or $dropout !~ /^[\.\d]+$/) {
      confess "Invalid dropout parameter '$dropout', must be number ";
   }
   $self->{dropout} = $dropout;

# switch these from Linear to arrays, embeddings x embeddings, otherwise the backprop doesn't work
   $self->{W_q} = defined($args{W_q})?dclone($args{W_q}):randn($self->{embeddings},$self->{embeddings});
   $self->{W_v} = defined($args{W_v})?dclone($args{W_v}):randn($self->{embeddings},$self->{embeddings});
   $self->{W_k} = defined($args{W_k})?dclone($args{W_k}):randn($self->{embeddings},$self->{embeddings});
   $self->{W_o} = defined($args{W_o})?dclone($args{W_o}):randn($self->{embeddings},$self->{embeddings});
   

   $self->{d_k} = int( $self->{embeddings} / $num_heads);

   if ($self->{embeddings} % $num_heads != 0) {
      die "embeddings must be divisible by num_heads";
   }

   # Adam optimiser state (first and second moment estimates for each weight matrix)
   my $e = $self->{embeddings};
   for my $key (qw(m_W_q v_W_q m_W_k v_W_k m_W_v v_W_v m_W_o v_W_o)) {
      $self->{$key} = [];
      for my $i (0 .. $e - 1) {
         for my $j (0 .. $e - 1) { $self->{$key}[$i][$j] = 0; }
      }
   }
   $self->{adam_beta1} = 0.9;
   $self->{adam_beta2} = 0.999;
   $self->{adam_epoch} = 1;

   return bless $self, $class;
}
sub forward {
   my $self = shift;
   my %args = @_;
# my $output = $mha->forward(q => $dummy_q, k => $dummy_k, v => $dummy_v, mask=>$dummy_mask_mha );
   say "ML::MHA forward called by " . join(", ", caller()) if $self->{debug};
   if ($args{batch}) {
      $args{q} = $args{batch};
      $args{v} = defined($args{batch_1})?$args{batch_1} : $args{batch};
      $args{k} = defined($args{batch_2})?$args{batch_2} : $args{batch};
   }
   $self->{input}             = dclone($args{q});
   $self->{input_kv}          = dclone($args{k});   # encoder output for cross-attn; same as input for self-attn
   $self->{is_cross_attention} = defined($args{batch_1}) ? 1 : 0;
   my $batch_size = scalar(@{$args{q}});
   my $seq_len = scalar(@{$args{q}->[0]});
   $self->{batch_size} = $batch_size;
   $self->{seq_len} = $seq_len;
   if ($self->{debug}) {
      say "ML::MHA batch size = $batch_size, seq_len = $seq_len";
   }
   my $Q = [];
   foreach my $b (0 .. $batch_size -1 ) {
      $Q->[$b] = matmul($args{q}->[$b], $self->{W_q});
      if ($self->{debug}) {
         print_2d_array("Q $b", $Q->[$b]);
      }
   }
   if ($self->{debug}) {
      say "Q = " . join(", ", scalar(@$Q), scalar(@{$Q->[0]}), scalar(@{$Q->[0][0]}));
   }
   my $K = [];
   foreach my $b (0 .. $batch_size -1 ) {
      $K->[$b] = matmul($args{v}->[$b], $self->{W_k});
      if ($self->{debug}) {
         print_2d_array("K $b", $K->[$b]);
      }
   }
   if ($self->{debug}) {
      say "K = " . join(", ", scalar(@$K), scalar(@{$K->[0]}), scalar(@{$K->[0][0]}));
   }
   my $V = [];
   foreach my $b (0 .. $batch_size -1 ) {
      $V->[$b] = matmul($args{v}->[$b], $self->{W_v});
      if ($self->{debug}) {
         print_2d_array("V $b", $V->[$b]);
      }
   }
   say "V = " . join(", ", scalar(@$V), scalar(@{$V->[0]}), scalar(@{$V->[0][0]})) if $self->{debug};
   # need to convert Q, K, V to [batch size, heads, seq_len, d_k]
   $self->{resized_Q} = []; # batch size x heads x seq len x d_k
   $self->{resized_K} = []; # batch size x heads x seq len x d_k
   $self->{resized_V} = []; # batch size x heads x seq len x d_k
   if ($self->{debug}) {
      say "MHA resizing input array from " . scalar(@$Q) . ", " . scalar(@{$Q->[0]}) . ", " . scalar(@{$Q->[0][0]})
          . " to $batch_size, " . $self->{num_heads} . ", " . $self->{seq_len} . ", " . $self->{d_k};
      say "MHA created by " . $self->{created_by};
   }
   foreach my $b (0 .. $batch_size - 1) {
      foreach my $h (0 .. $self->{num_heads} - 1) {
         foreach my $i (0 .. $self->{seq_len} - 1) {
            foreach my $d (0 .. $self->{d_k} - 1) {
               $self->{resized_Q}->[$b][$h][$i][$d] = $Q->[$b][$i][($h * $self->{d_k} ) + $d];
               $self->{resized_K}->[$b][$h][$i][$d] = $K->[$b][$i][($h * $self->{d_k} ) + $d];
               $self->{resized_V}->[$b][$h][$i][$d] = $V->[$b][$i][($h * $self->{d_k} ) + $d];
            }
         }
         if ($self->{debug}) {
            print_2d_array("resized K $b $h", $self->{resized_K}->[$b][$h]);
            print_2d_array("resized V $b $h", $self->{resized_V}->[$b][$h]);
            print_2d_array("resized Q $b $h", $self->{resized_Q}->[$b][$h]);
         }
      }
   }
   say "resized Q, V, K  shape: " . join(", ", $batch_size, $self->{num_heads}, $self->{seq_len}, $self->{d_k}) if $self->{debug};
   my ($output, $attention_weights) = scaled_dot_product_attention($self->{resized_Q}, $self->{resized_K}, $self->{resized_V}, $args{mask}, $self->{dropout});
   if ($self->{debug}) {
      say "output shape = " . join(", ", scalar(@$output), scalar(@{$output->[0]}), scalar(@{$output->[0][0]}), scalar(@{$output->[0][0][0]}));
      foreach my $b (0 .. $batch_size - 1) {
         foreach my $h (0 .. $self->{num_heads} - 1) {
            print_2d_array("out heads $b $h", $output->[$b][$h]);
         }
      }
   }
   my $resized_output = [];
   foreach my $b (0 .. $batch_size - 1) {
      foreach my $h (0 .. $self->{num_heads} - 1) {
         foreach my $i (0 .. $self->{seq_len} - 1) {
            foreach my $d (0 .. $self->{d_k} - 1) {
               $resized_output->[$b][$i][($h * $self->{d_k} ) + $d] = $output->[$b][$h][$i][$d];
            }
         }
      }
      print_2d_array("concat $b", $resized_output->[$b]) if $self->{debug};
   }
   $self->{concat} = dclone($resized_output); # needed for backprop
   say "resized output shape = " . join(", ", scalar(@$resized_output), scalar(@{$resized_output->[0]}), scalar(@{$resized_output->[0][0]})) if $self->{debug};
   $output = [];
   foreach my $b (0 .. $batch_size -1 ) {
      #$output->[$b] = $self->{W_o}->forward(batch => $resized_output->[$b]);
      print_2d_array("resized output (concat) $b", $resized_output->[$b]) if $self->{debug};
      $output->[$b] = matmul($resized_output->[$b], $self->{W_o});
      print_2d_array("final output $b", $output->[$b]) if $self->{debug};
   }
   say "output shape after linear= " . join(", ", scalar(@$output), scalar(@{$output->[0]}), scalar(@{$output->[0][0]})) if $self->{debug};
   say "attention_weights shape = " . join(", ", scalar(@$attention_weights), scalar(@{$attention_weights->[0]}), scalar(@{$attention_weights->[0][0]}), scalar(@{$attention_weights->[0][0][0]})) if $self->{debug};
   $self->{attention_weights} = dclone($attention_weights);
   return $output;
}

sub backward {
   my $self = shift;
   my %params = @_;
   my $next = $params{ next };
   my $in_grad = $next->gradient();
   $self->{weights_grad} = dclone($in_grad);
   # need to transpose the last 2 dimensions of our output
   # the original python didn't seem to work, the shape of dWo should be embeddings x embeddings, i.e. the same as W_o.
   # changed the code to do the same as for dWk, dWv and dWk
   # new version of dWo
   #foreach my $e1 (0 .. $self->{embeddings} - 1) {
      #foreach my $e2 (0 .. $self->{embeddings} - 1) {
         #$self->{dWo}[$e1][$e2] = 0;
      #}
   #}

   foreach my $b (0 .. $self->{batch_size} - 1) {
      $self->{d_concat}[$b] =  matmul( $in_grad->[$b], transpose($self->{W_o}));
      print_2d_array("d_concat ", $self->{d_concat}[$b]) if $self->{debug};
   }

   my $concat_reshape = [];
   my $dout_reshape = [];
   foreach my $b (0 .. $self->{batch_size} - 1) {
      foreach my $s(0 .. $self->{seq_len} - 1) {
         foreach my $e(0 .. $self->{embeddings} - 1) {
            $concat_reshape->[$b * $self->{seq_len} + $s][$e] = $self->{concat}[$b][$s][$e];
            $dout_reshape->[$b * $self->{seq_len} + $s][$e] = $in_grad->[$b][$s][$e];
         }
      }
   }
   print_2d_array("concat_reshape", $concat_reshape) if $self->{debug};
   print_2d_array("dout", $dout_reshape) if $self->{debug};
   $self->{dWo} = matmul(transpose( $concat_reshape ), $dout_reshape );

   print_2d_array("dWo", $self->{dWo}) if $self->{debug};
   
   #my $WoT = transpose($self->{W_o});

   #my $d_concat = []; # batch size x seq length x embeddings
   #foreach my $b (0 .. $self->{batch_size} - 1) {
      #$d_concat->[$b] = matmul($self->{weights_grad}[$b], $WoT);
   #}

   # d_concat = batch x seqlen x embeddings, convert to d_out_heads which is batch_size x seqlen x n_heads x d_k
   my $d_out_heads = [];
   foreach my $b (0 .. $self->{batch_size} - 1) {
      foreach my $n (0 .. $self->{num_heads} - 1) {
         foreach my $s (0 .. $self->{seq_len} - 1) {
            foreach my $d (0 .. $self->{d_k} - 1) {
               $d_out_heads->[$b][$n][$s][$d] = $self->{d_concat}->[$b][$s][$n * $self->{d_k} + $d];
            }
         }
      }
   }
   my $d_attn = [];
   foreach my $b (0 .. $self->{batch_size} - 1) {
      foreach my $n (0 .. $self->{num_heads} - 1) {
         $d_attn->[$b][$n] = matmul($d_out_heads->[$b][$n], transpose($self->{resized_V}[$b][$n]));
         if ($self->{debug}) {
            print_2d_array("d_out_heads $b $n",  $d_out_heads->[$b][$n]);
            print_2d_array("transposed resized V $b $n", transpose($self->{resized_V}[$b][$n]));
            print_2d_array("d_attn $b $n", $d_attn->[$b][$n]);
         }
      }
   }

   # calculate dV from attention_weights x d_out_heads

   my $dVs = [];
   foreach my $b (0 .. $self->{batch_size} - 1) {
      foreach my $n (0 .. $self->{num_heads} - 1) {
         $dVs->[$b][$n] = matmul(transpose($self->{attention_weights}[$b][$n]), $d_out_heads->[$b][$n]);
         if ($self->{debug}) {
            print_2d_array("dVs $b $n", $dVs->[$b][$n]);
         }
      }
   }    
   my $dA = []; # batch x heads x seqlen x seqlen
   foreach my $b (0 .. $self->{batch_size} - 1) {
      foreach my $n (0 .. $self->{num_heads} - 1) {
         $dA->[$b][$n] = matmul($d_out_heads->[$b][$n], transpose($self->{resized_V}->[$b][$n]));
      }
   }    
   
   my $d_score_factor = sqrt( $self->{d_k} );

   my $AdA = [];
   foreach my $b (0 .. $self->{batch_size} - 1) {
      foreach my $n (0 .. $self->{num_heads} - 1) {
         my $I = mult_2_arrays($dA->[$b][$n], $self->{attention_weights}->[$b][$n]);
         # need to sum each row
         foreach my $s (0 .. $self->{seq_len} - 1) {
            $AdA->[$b][$n][$s][0] = 0;
            foreach my $s2 (0 .. $self->{seq_len} - 1) {
               $AdA->[$b][$n][$s][0] += $I->[$s][$s2];
            }
         }
      }
   }

   my $d_scores = [];
   foreach my $b (0 .. $self->{batch_size} - 1) {
      foreach my $n (0 .. $self->{num_heads} - 1) {
         foreach my $s (0 .. $self->{seq_len} - 1) {
            foreach my $s2 (0 .. $self->{seq_len} - 1) {
               $d_scores->[$b][$n][$s][$s2] = ( $self->{attention_weights}->[$b][$n][$s][$s2] * ( $dA->[$b][$n][$s][$s2] - $AdA->[$b][$n][$s][0] ) ) / $d_score_factor;
            }
         }
         print_2d_array("d_scores $b $n", $d_scores->[$b][$n]) if $self->{debug};
      }
   }
   my $dQs = [];
   foreach my $b (0 .. $self->{batch_size} - 1) {
      foreach my $n (0 .. $self->{num_heads} - 1) {
         $dQs->[$b][$n] = matmul( $d_scores->[$b][$n], $self->{resized_K}[$b][$n]);
         print_2d_array("dQs (d_Qh) $b $n", $dQs->[$b][$n] ) if $self->{debug};
      }
   }
   my $dKs = [];
   foreach my $b (0 .. $self->{batch_size} - 1) {
      foreach my $n (0 .. $self->{num_heads} - 1) {
         $dKs->[$b][$n] = matmul( transpose($d_scores->[$b][$n]), $self->{resized_Q}[$b][$n] );
         print_2d_array("dKs (d_Kh) $b $n", $dKs->[$b][$n] ) if $self->{debug};
      }
   }
   my $dQ = []; # contains combined head: batch x seq len x embeddings
   foreach my $b (0 .. $self->{batch_size} - 1) {
      foreach my $n (0 .. $self->{num_heads} - 1) {
         foreach my $s (0 .. $self->{seq_len} - 1) {
            foreach my $d (0 .. $self->{d_k} - 1) {
               $dQ->[$b][$s][$n * $self->{d_k} + $d] = $dQs->[$b][$n][$s][$d];
            }
         }
      }
      print_2d_array("dQ $b", $dQ->[$b]) if $self->{debug};
   }
   my $dV = []; # contains combined head: batch x seq len x embeddings
   foreach my $b (0 .. $self->{batch_size} - 1) {
      foreach my $n (0 .. $self->{num_heads} - 1) {
         foreach my $s (0 .. $self->{seq_len} - 1) {
            foreach my $d (0 .. $self->{d_k} - 1) {
               $dV->[$b][$s][$n * $self->{d_k} + $d] = $dVs->[$b][$n][$s][$d];
            }
         }
      }
      print_2d_array("dV $b", $dV->[$b]) if $self->{debug};
   }
   my $dK = []; # contains combined head: batch x seq len x embeddings
   foreach my $b (0 .. $self->{batch_size} - 1) {
      foreach my $n (0 .. $self->{num_heads} - 1) {
         foreach my $s (0 .. $self->{seq_len} - 1) {
            foreach my $d (0 .. $self->{d_k} - 1) {
               $dK->[$b][$s][$n * $self->{d_k} + $d] = $dKs->[$b][$n][$s][$d];
            }
         }
      }
      print_2d_array("dK $b", $dK->[$b]) if $self->{debug};
   }
   $self->{dWq} = []; # embedding x batch size x embedding 
   $self->{dWk} = []; # embedding x batch size x embedding 
   $self->{dWv} = []; # embedding x batch size x embedding 
# as with dWo, above, do a matmul per batch and add.
   foreach my $e1 (0 .. $self->{embeddings} - 1) {
      foreach my $e2 (0 .. $self->{embeddings} - 1) {
         $self->{dWq}[$e1][$e2] = 0;
         $self->{dWk}[$e1][$e2] = 0;
         $self->{dWv}[$e1][$e2] = 0;
      }
   }
   foreach my $b (0 .. $self->{batch_size} - 1) {
       $self->{dWq} = add_2_arrays($self->{dWq}, matmul(transpose($self->{input}[$b]),    $dQ->[$b]));
       $self->{dWk} = add_2_arrays($self->{dWk}, matmul(transpose($self->{input_kv}[$b]), $dK->[$b]));
       $self->{dWv} = add_2_arrays($self->{dWv}, matmul(transpose($self->{input_kv}[$b]), $dV->[$b]));
   }
   if ($self->{debug}) {
      print_2d_array("dWq", $self->{dWq});
      print_2d_array("dWk", $self->{dWk});
      print_2d_array("dWv", $self->{dWv});
   }

   my $dX_q = [];
   my $dX_k = [];
   my $dX_v = [];
   my $dX    = [];
   my $dX_kv = [];
   foreach my $b (0 .. $self->{batch_size} - 1) {
      foreach my $s (0 .. $self->{seq_len} - 1) {
         foreach my $e (0 .. $self->{embeddings} - 1) {
            $dX->[$b][$s][$e]    = 0;
            $dX_kv->[$b][$s][$e] = 0;
         }
      }
   }
   foreach my $b (0 .. $self->{batch_size} - 1) {
      $dX_q->[$b] = matmul( $dQ->[$b], transpose( $self->{W_q} ) );
      print_2d_array("dX_q $b", $dX_q->[$b]) if $self->{debug};
      $dX->[$b] = add_2_arrays($dX->[$b], $dX_q->[$b]);
      $dX_k->[$b] = matmul( $dK->[$b], transpose( $self->{W_k} ) );
      print_2d_array("dX_k $b", $dX_k->[$b]) if $self->{debug};
      $dX_v->[$b] = matmul( $dV->[$b], transpose( $self->{W_v} ) );
      print_2d_array("dX_v $b", $dX_v->[$b]) if $self->{debug};
      if ($self->{is_cross_attention}) {
         # K/V source is encoder output — its gradient is tracked separately
         $dX_kv->[$b] = add_2_arrays($dX_kv->[$b], $dX_k->[$b]);
         $dX_kv->[$b] = add_2_arrays($dX_kv->[$b], $dX_v->[$b]);
      } else {
         $dX->[$b] = add_2_arrays($dX->[$b], $dX_k->[$b]);
         $dX->[$b] = add_2_arrays($dX->[$b], $dX_v->[$b]);
      }
      print_2d_array("dX $b", $dX->[$b]) if $self->{debug};
   }
   $self->{gradient}    = dclone($dX);
   $self->{gradient_kv} = dclone($dX_kv);
}

sub gradient {
   my $self = shift;
   return $self->{gradient};
}

sub gradient_kv {
   my $self = shift;
   return $self->{gradient_kv};
}

sub optimise {
   my $self = shift;
   my %args = @_;
   my $lr = $args{learning_rate} || 0.001;
   my $t  = $self->{adam_epoch};
   adam_optimiser($self->{dWo}, $self->{m_W_o}, $self->{v_W_o}, $self->{W_o}, $lr, $self->{adam_beta1}, $self->{adam_beta2}, $t);
   adam_optimiser($self->{dWq}, $self->{m_W_q}, $self->{v_W_q}, $self->{W_q}, $lr, $self->{adam_beta1}, $self->{adam_beta2}, $t);
   adam_optimiser($self->{dWk}, $self->{m_W_k}, $self->{v_W_k}, $self->{W_k}, $lr, $self->{adam_beta1}, $self->{adam_beta2}, $t);
   adam_optimiser($self->{dWv}, $self->{m_W_v}, $self->{v_W_v}, $self->{W_v}, $lr, $self->{adam_beta1}, $self->{adam_beta2}, $t);
   if ($self->{debug}) {
      print_2d_array("W_o", $self->{W_o});
      print_2d_array("W_q", $self->{W_q});
      print_2d_array("W_v", $self->{W_v});
      print_2d_array("W_k", $self->{W_k});
   }
   $self->{dWo} = [];
   $self->{dWq} = [];
   $self->{dWv} = [];
   $self->{dWk} = [];
   $self->{adam_epoch}++;
}

1;

