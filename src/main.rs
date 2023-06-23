use clap::Parser;
use std::path::PathBuf;
use tokenizers::models::wordpiece::{WordPiece, WordPieceTrainer};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, TokenizerBuilder};

#[derive(Parser)]
struct Cli {
    /// vocabulary size
    #[clap(short, long)]
    size: usize,

    /// save path
    #[clap(short, long)]
    out: PathBuf,

    /// list of input txt text files
    #[clap(short, long, value_parser, num_args = 1.., value_delimiter = ' ')]
    txt: Vec<String>,
}

fn main() {
    let cli = Cli::parse();

    let mut trainer = WordPieceTrainer::builder()
        .vocab_size(cli.size)
        .show_progress(true)
        .special_tokens(vec![
            AddedToken::from(String::from("[PAD]"), true),
            AddedToken::from(String::from("[UNK]"), true),
            AddedToken::from(String::from("[CLS]"), true),
            AddedToken::from(String::from("[SEP]"), true),
            AddedToken::from(String::from("[MASK]"), true),
        ])
        .build();

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(WordPiece::default())
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            NFC.into(),
        ])))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()
        .expect("Failed to build tokenizer");

    tokenizer
        .train_from_files(&mut trainer, cli.txt)
        .expect("Failed to train")
        .save(cli.out, false)
        .expect("Failed to save");
}
