use thiserror::Error;

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("CSV parse error: {0}")]
    Csv(#[from] csv::Error),

    #[error("invalid graph: {0}")]
    Invalid(String),

    #[error("unknown compression method: {0}")]
    UnknownMethod(String),

    #[error("requested K={requested} larger than available nodes ({available})")]
    KTooLarge { requested: usize, available: usize },

    #[error("zenodo parse: {0}")]
    Zenodo(String),
}

pub type GraphResult<T> = Result<T, GraphError>;
