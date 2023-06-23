use clap::Parser;
use rocksdb::{Cache, DBRawIteratorWithThreadMode, Env, Options, ReadOptions, DB};
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

    #[clap(short, long)]
    db: Option<PathBuf>,

    /// list of input txt text files
    #[clap(short, long, value_parser, num_args = 1.., value_delimiter = ' ')]
    txt: Vec<String>,
}

struct ValueIter<'a> {
    inner: DBRawIteratorWithThreadMode<'a, DB>,
    key_count_estimate: usize,
    first: bool,
}

impl<'a> ValueIter<'a> {
    fn new(db: &'a DB, key_count_estimate: usize) -> Self {
        let mut read_opt = ReadOptions::default();
        read_opt.set_async_io(true);
        let mut inner = db.raw_iterator_opt(read_opt);
        inner.seek_to_first();
        Self {
            inner,
            key_count_estimate,
            first: true,
        }
    }
}

impl<'a> Iterator for ValueIter<'a> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        if !self.first {
            self.inner.next();
        } else {
            self.first = false;
        }
        self.inner
            .value()
            .map(|b| String::from_utf8_lossy(b).to_string())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.key_count_estimate))
    }
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

    if let Some(db) = cli.db {
        let env = Env::new().expect("failed to create env");
        let cache = Cache::new_lru_cache(1024 * 1024 * 8);
        let (opts, cfs) =
            Options::load_latest(&db, env, true, cache).expect("failed to load cache");
        let db =
            DB::open_cf_descriptors_read_only(&opts, db, cfs, false).expect("Failed to open db");
        let estimate_key = db
            .property_int_value("rocksdb.estimate-num-keys")
            .expect("failed to get estimate key num")
            .unwrap() as usize;
        let value_iter = ValueIter::new(&db, estimate_key);
        tokenizer
            .train(&mut trainer, value_iter)
            .expect("Failed to train")
            .save(cli.out, false)
            .expect("Failed to save");
    } else {
        tokenizer
            .train_from_files(&mut trainer, cli.txt)
            .expect("Failed to train")
            .save(cli.out, false)
            .expect("Failed to save");
    }
}
