use std::fmt;
use std::slice;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Path(Vec<PathComponent>);

// Names of definitions
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PathComponent(String); // only [a-zA-Z_]

pub fn path(components: Vec<&str>) -> Result<Path, InvalidPathComponent> {
    components.try_into()
}

impl Path {
    pub fn iter(&self) -> slice::Iter<'_, PathComponent> {
        self.0.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn concat(&self, other: &Path) -> Path {
        let mut components = self.0.clone();
        components.extend(other.0.clone());
        Path(components)
    }

    pub fn push(&self, component: &str) -> Option<Path> {
        let mut components = self.0.clone();
        components.push(component.to_string().try_into().ok()?);
        Some(Path(components))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Display/TryFrom instances

#[derive(Debug, Clone, PartialEq)]
pub struct InvalidPathComponent(pub String);

impl TryFrom<String> for PathComponent {
    type Error = InvalidPathComponent;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.chars().all(|c| c.is_alphanumeric() || c == '_') {
            Ok(PathComponent(value))
        } else {
            Err(InvalidPathComponent(value))
        }
    }
}

impl TryFrom<Vec<&str>> for Path {
    type Error = InvalidPathComponent;

    fn try_from(value: Vec<&str>) -> Result<Self, Self::Error> {
        let components: Result<Vec<PathComponent>, InvalidPathComponent> = value
            .into_iter()
            .map(|s| s.to_string().try_into())
            .collect();
        Ok(Path(components?))
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
