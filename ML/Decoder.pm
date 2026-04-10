package ML::Decoder;

use Modern::Perl;
use ML::LayerNormalisation;
use Data::Dumper;
use Storable qw(dclone);
use ML::DecoderBlock;

use Cwd qw(abs_path);
use Math::Random qw(random_uniform);
use ML::Util qw(print_2d_array add_2_arrays rotate_matrix_180);

sub type {
   return "ML::Decoder";
}

sub initialise {
   my $self = shift;
}

sub optimise {
   my $self = shift;
   foreach my $l (0 .. $self->{layers} - 1) {
      $self->{DecoderBlocks}[$l]->optimise( @_ );
   }
   $self->{norm}->optimise( @_ );
}

sub forward {
   my $self = shift;
   my %args = @_;
   $self->{input} = dclone($args{batch});
   $self->{batch_size} = scalar(@{$self->{input}});

   my $output = [];
   foreach my $l (0 .. $self->{layers} - 1) {
      say "ML::Decoder->forward calling layer_idx $l" if $self->{debug};
      $self->{input} = $self->{DecoderBlocks}[$l]->forward( batch => $self->{input}, 
                                                            encoder_output => $args{encoder_output}, 
                                                            src_mask => $args{src_mask}, 
                                                            tgt_mask => $args{tgt_mask} );
   }
   $output = $self->{norm}->forward( batch => $self->{input} );

   return $output;
}
 
sub backward {
   my $self = shift;
   my %params = @_;
   
   $self->{batch_size} ||= $params{batch_size};
 
   my $next = $params{ next };
   $self->{norm}->backward( next => $next );
   $next = $self->{norm};
   if ($self->{debug}) {
      foreach my $b ( 0 .. $self->{batch_size} - 1) {
         print_2d_array("Decoder backward grad after norm $b", $self->{norm}->{gradient}[$b]);
      }
   }
   my $layer_idx = $self->{layers} - 1;

   foreach my $l (reverse @{$self->{DecoderBlocks}}) {
      $l->backward( next => $next );
      if ($self->{debug}) {
         foreach my $b ( 0 .. $self->{batch_size} - 1) {
            print_2d_array("Decoder backward grad after layer $layer_idx $b ", $l->{gradient}[$b]);
         }
      }
      $layer_idx--;
      $next = $l;
   }
   $self->{gradient} = $self->{DecoderBlocks}[0]->gradient();
}

sub gradient {
   my $self = shift;
   return $self->{gradient};
}

sub gradient_enc {
   my $self = shift;
   my $grad = undef;
   foreach my $block (@{$self->{DecoderBlocks}}) {
      my $g = $block->gradient_enc();
      next unless defined $g;
      if (!defined $grad) {
         $grad = dclone($g);
      } else {
         for my $b (0 .. $#{$g}) {
            $grad->[$b] = add_2_arrays($grad->[$b], $g->[$b]);
         }
      }
   }
   return $grad;
}

sub outsize {
   my $self = shift;
   return $self->{insize};
}

sub name {
   my $self = shift;
   return $self->{name};
}

sub get_weights {
   my $self = shift;
   my $weights = { embeddings => $self->{embeddings},
                   max_seq_length => $self->{max_seq_length},
                   tgt_seq_length => $self->{tgt_seq_length},
                   layers => $self->{layers},
                   num_heads => $self->{num_heads},
                   d_ff => $self->{d_ff},
                   dropout => $self->{dropout},
                   normalisation => $self->{norm}->get_weights(),
                 };
   foreach my $i (0 .. $self->{layers} - 1) {
      push @{$weights->{decoderblocks}}, $self->{DecoderBlocks}[$i]->get_weights();
   }
   return $weights;
}

sub new {
   my $class = shift;
   my %args = @_;

   my $embeddings = $args{embeddings};
   die "Decoder invalid embeddings: " . $args{embeddings} unless defined($embeddings) and $embeddings =~ /^\d+$/;
   my $max_seq_length = $args{max_seq_length};
   die "Decoder invalid max_seq_length: " . $args{max_seq_length} unless defined($max_seq_length) and $max_seq_length =~ /^\d+$/;
   my $tgt_seq_length = $args{tgt_seq_length};
   die "Decoder invalid tgt_seq_length: " . $args{tgt_seq_length} unless defined($tgt_seq_length) and $tgt_seq_length =~ /^\d+$/;
   my $num_heads = $args{num_heads};
   die "Decoder invalid num_heads: " . $args{num_heads} unless defined($num_heads) and $num_heads =~ /^\d+$/;
   my $d_ff = $args{d_ff};
   die "Decoder invalid d_ff: " . $args{d_ff} unless defined($d_ff) and $d_ff =~ /^\d+$/;
   my $dropout = $args{dropout};
   die "Decoder invalid dropout: " . $args{dropout} unless defined($dropout) and $dropout =~ /^\d+\.?\d+?$/;
   my $self = {};
   $self->{dropout} = $dropout;
   $self->{embeddings} = $embeddings;
   $self->{max_seq_length} = $max_seq_length;
   $self->{tgt_seq_length} = $tgt_seq_length;
   die "Decoder parameter layers  missing\n" unless defined($args{layers});
   $self->{layers} = $args{layers};
   $self->{num_heads} = $args{num_heads};
   $self->{d_ff} = $args{d_ff};
   $self->{DecoderBlocks} = [];
   foreach my $i (0 .. $self->{layers} - 1) {
      if (defined($args{decoderblocks}->[$i])) {
         push @{$self->{DecoderBlocks}}, ML::DecoderBlock->new( %{ $args{decoderblocks}->[$i] } );
      } else {
         push @{$self->{DecoderBlocks}}, ML::DecoderBlock->new( embeddings => $self->{embeddings}, 
                                                             max_seq_length => $self->{max_seq_length}, 
                                                             dropout => $self->{dropout}, 
                                                             num_heads => $self->{num_heads},
                                                             d_ff => $self->{d_ff},
                                                             dropout => $self->{dropout},
                                                             tgt_seq_length => $self->{tgt_seq_length} );
      }
   }

   $self->{created_by} = "ML::Decoder created by " . join(", ", caller()) . " with tgt_seq_length $tgt_seq_length";
   if (defined($args{normalisation})) {
      $self->{norm} = ML::LayerNormalisation->new( embeddings => $args{normalisation}->{embeddings}, max_seq_length => $args{normalisation}->{max_seq_length});
      $self->{norm}->initialise( weights => { gamma => $args{normalisation}->{gamma}, beta => $args{normalisation}->{beta} } );
   } else {
      $self->{norm} = ML::LayerNormalisation->new( embeddings => $self->{embeddings}, 
                                                max_seq_length => $self->{max_seq_length}, 
                                                creator => $self->{created_by});
      $self->{norm}->initialise();
   }
   $self->{debug} = $args{debug};
   $self->{debug} ||= 0;
   $self->{name} = $args{name} || "decoder";
   return bless $self, $class;
}  

1;
