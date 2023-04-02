use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use ordered_float::OrderedFloat;

#[derive(Debug, Clone)]
pub struct Cache {
    cache: HashMap<FloatArrayKey, f32>,
}

impl Cache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub fn get(&self, key: [[[f32; 8]; 8]; 13]) -> Option<f32> {
        self.cache.get(&FloatArrayKey(key)).cloned()
    }

    pub fn insert(&mut self, key: [[[f32; 8]; 8]; 13], value: f32) {
        self.cache.insert(FloatArrayKey(key), value);
    }
}

#[derive(Debug, Clone)]
struct FloatArrayKey([[[f32; 8]; 8]; 13]);

impl PartialEq for FloatArrayKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0.iter()).all(|(a_mat, b_mat)| {
            a_mat.iter().zip(b_mat.iter()).all(|(a_row, b_row)| {
                a_row.iter().zip(b_row.iter()).all(|(a, b)| OrderedFloat(*a) == OrderedFloat(*b))
            })
        })
    }
}

impl Eq for FloatArrayKey {}

impl Hash for FloatArrayKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // for mat in &self.0 {
        //     for row in mat {
        //         for value in row {
        //             OrderedFloat(*value).hash(state);
        //         }
        //     }
        // }
        self.0.map(|mat| mat.map(|row| row.map(|value| OrderedFloat(value)))).hash(state);
    }
}   