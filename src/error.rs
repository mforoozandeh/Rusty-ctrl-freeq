//! The crate's error type.
//!
//! The Python original raises exceptions from deep inside its setup code.  Here every failure is a typed value that
//! propagates through [`Result`], so a caller - the GUI in particular - can show it instead of crashing.

use std::fmt;

/// Everything that can go wrong inside ctrl-freeq.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The configuration is missing a value, has an inconsistent one, or names something unknown.
    Config(String),
    /// A valid request for something this version does not implement, such as an algorithm name from the Python
    /// package that has not been ported.
    NotSupported(String),
    /// Two operands have shapes that do not line up.
    Dimension(String),
    /// A numerical routine failed or produced a non-finite value.
    Numerical(String),
    /// Reading or writing a file failed.
    Io(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config(s) => write!(f, "invalid configuration: {s}"),
            Error::NotSupported(s) => write!(f, "not supported: {s}"),
            Error::Dimension(s) => write!(f, "dimension mismatch: {s}"),
            Error::Numerical(s) => write!(f, "numerical failure: {s}"),
            Error::Io(s) => write!(f, "io error: {s}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
