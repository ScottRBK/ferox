use std::fs::File;
use std::io::{ Write, Read, ErrorKind } ;
use std::path::Path;
use std::error::Error;
use serde:: {Serialize, Deserialize};

#[derive(Deserialize, Serialize, Debug)]
#[serde(default)]
pub struct Settings {
    pub service_name: String,
    pub log_level: LogLevel,
}

static SETTINGS_FILE_NAME: &str = "ferox.toml";

 impl Settings {
    pub fn new() -> Settings {
        let settings = Settings::from_path(SETTINGS_FILE_NAME);
        match settings {
            Ok(settings) => settings,
            Err(e) => panic!["Error loading file {e}"],
        }
    }

    pub fn generate_file() -> Result<(), Box<dyn Error>>{
        let mut settings_file = File::create_new(SETTINGS_FILE_NAME)?;
        let settings_str = toml::to_string_pretty::<Settings>(&Settings::default())?;
        settings_file.write_all(settings_str.as_bytes())?;
        Ok(()) 
    }

    fn from_path(path: &str) -> Result<Settings, Box<dyn Error>> {

        let settings_path = Path::new(path);
        let mut settings_file = match File::open(&settings_path) {
            Ok(file) => file,
            Err(e) if e.kind() == ErrorKind::NotFound => {
                Self::generate_file()?;
                return Ok(Settings::default());
            },
            Err(e) => return Err(e.into()),
        };

        let mut settings_str = String::new();
        settings_file.read_to_string(&mut settings_str)?;
        Self::from_str(&settings_str)
    }

    fn from_str(settings_str: &str) -> Result<Settings, Box<dyn Error>> {

        toml::from_str::<Settings>(&settings_str).map_err(Into::into)
    }
}

impl Default for Settings {

    fn default() -> Settings {
        Settings {
            service_name: "Ferox".to_string(),
            log_level: LogLevel::Debug,
        } 
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
}
