use std::fmt;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Path(Vec<PathComponent>);

// Names of definitions
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PathComponent(String); // only [a-zA-Z_]

pub fn path(components: Vec<&str>) -> Path {
    components.try_into().expect("invalid path")
}

////////////////////////////////////////////////////////////////////////////////
// Display/TryFrom instances

impl TryFrom<String> for PathComponent {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.chars().all(|c| c.is_alphanumeric() || c == '_') {
            Ok(PathComponent(value))
        } else {
            Err(format!(
                "PathComponent must only contain alphanumeric characters and underscores, got: {value}"
            ))
        }
    }
}

impl TryFrom<Vec<&str>> for Path {
    type Error = String;

    fn try_from(value: Vec<&str>) -> Result<Self, Self::Error> {
        let components: Result<Vec<PathComponent>, String> = value
            .into_iter()
            .map(|s| s.to_string().try_into())
            .collect();
        components.map(Path)
    }
}

impl fmt::Display for PathComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let components: Vec<String> = self.0.iter().map(|c| c.to_string()).collect();
        write!(f, "{}", components.join("."))
    }
}
