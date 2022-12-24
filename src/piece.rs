use std::{
    any::{Any, TypeId},
    fmt::{Display},
};

// use colored::{Colorize};
use crate::game::Game;
use crate::player::Player;

//should I wrap legal moves in a enum?

pub trait Piece: Display + DynClone {
    fn valid_move(&self, from: (u8, u8), to: (u8, u8), game: &Game) -> Move;
    fn get_legal_moves(&self, position: (u8, u8), game: &Game) -> Vec<(u8, u8)>;
    fn player(&self) -> Player;
    fn can_intercept_path(
        &self,
        position: (u8, u8),
        enemy: (u8, u8),
        king: (u8, u8),
        game: &Game,
    ) -> Vec<(u8, u8)> {
        let moves = self.get_legal_moves(position, game);
        let mut intercepts = Vec::new();
        let (x, y) = (king.0 as i8 - enemy.0 as i8, king.1 as i8 - enemy.1 as i8);
        let (x_sign, y_sign) = (x.signum(), y.signum());
        if x != 0 && y != 0 && x.abs() != y.abs() {
            return intercepts;
        }
        let (mut j, mut i) = enemy;
        if moves.contains(&(j, i)) {
            intercepts.push((j, i));
        }
        while j != king.0 && i != king.1 {
            j = (j as i8 + x_sign) as u8;
            i = (i as i8 + y_sign) as u8;
            if moves.contains(&(j, i)) {
                intercepts.push((j, i));
            }
        }
        intercepts
    }
    fn as_any(&self) -> &dyn Any;
    fn name(&self) -> &str;
}

pub trait DynClone {
    fn clone_box(&self) -> Box<dyn Piece>;
}

pub trait Construct {
    fn new(player: Player) -> Self;
}

impl Clone for Box<dyn Piece> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl dyn Piece {
    pub fn is_type<T: Piece + 'static>(&self) -> bool {
        self.as_any().type_id() == TypeId::of::<T>()
    }

    pub fn get_piece<T: Piece + 'static>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }

    pub fn new_piece<T: Piece + Construct + 'static>(player: Player) -> Box<Self> {
        Box::new(T::new(player))
    }

    pub fn display<T: Piece + Display + 'static>(piece: &T) -> String {
        format!("{piece}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Move {
    Normal,
    Double((u8, u8)),
    Castle,
    EnPassant((u8, u8)),
    Invalid,
    // Invalid(friendly: bool)?
}