use thiserror::Error;

/// Result alias used across all flybrain crates.
pub type FlybrainResult<T> = Result<T, FlybrainError>;

#[derive(Debug, Error)]
pub enum FlybrainError {
    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("budget exceeded: {0}")]
    BudgetExceeded(String),

    #[error("invalid graph action: {0}")]
    InvalidAction(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("io error: {0}")]
    Io(String),
}

impl From<std::io::Error> for FlybrainError {
    fn from(e: std::io::Error) -> Self {
        FlybrainError::Io(e.to_string())
    }
}
