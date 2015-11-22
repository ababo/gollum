use std::io;
use std::fs;
use rustc_serialize::json;
use std::io::prelude::*;

#[derive(Debug)]
pub enum Error {
	JsonError(json::DecoderError),
	IoError(io::Error)
}

impl From<json::DecoderError> for Error {
    fn from(err: json::DecoderError) -> Self {
    	Error::JsonError(err)
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
    	Error::IoError(err)
    }
}

#[derive(RustcDecodable)]
pub struct Config {
	pub cblas_dylib_path: String,
}

impl Config {
	pub fn new() -> Result<Config, Error> {
		let mut f = try!(fs::File::open("config.json"));
		let mut s = String::new();
		try!(f.read_to_string(&mut s));
		Ok(try!(json::decode(&s)))
	}
}
