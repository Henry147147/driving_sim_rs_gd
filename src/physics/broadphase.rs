//! Godot 4.3 BVH leaf ordering, splitting and incremental optimization.
//! Adapted from core/math/bvh*. See GODOT_LICENSE for the upstream license.
use crate::math::godot_math::F2;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Aabb {
    pub(crate) min: F2,
    pub(crate) max: F2,
}
impl Aabb {
    pub(crate) fn from_rect(position: F2, size: F2) -> Self {
        Self {
            min: position,
            max: position + size,
        }
    }
    pub(crate) fn size(self) -> F2 {
        self.max - self.min
    }
    pub(crate) fn grow(self, margin: f32) -> Self {
        let d = F2 { x: margin, y: margin };
        Self {
            min: self.min - d,
            max: self.max + d,
        }
    }
    fn merge(self, other: Self) -> Self {
        Self {
            min: F2 {
                x: self.min.x.min(other.min.x),
                y: self.min.y.min(other.min.y),
            },
            max: F2 {
                x: self.max.x.max(other.max.x),
                y: self.max.y.max(other.max.y),
            },
        }
    }
    pub(crate) fn encloses(self, other: Self) -> bool {
        other.min.x >= self.min.x && other.min.y >= self.min.y && other.max.x <= self.max.x && other.max.y <= self.max.y
    }
    pub(crate) fn intersects(self, other: Self) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x && self.min.y <= other.max.y && self.max.y >= other.min.y
    }
    fn proximity(self, other: Self) -> f32 {
        let d = (self.min + self.max) - (other.min + other.max);
        d.x.abs() + d.y.abs()
    }
    fn empty() -> Self {
        Self {
            min: F2 {
                x: f32::MAX,
                y: f32::MAX,
            },
            max: F2 {
                x: -f32::MAX,
                y: -f32::MAX,
            },
        }
    }
    pub(crate) fn roundtrip_rect(self) -> Self {
        Self::from_rect(self.min, self.size())
    }
}

#[derive(Clone)]
struct Node {
    bounds: Aabb,
    parent: Option<usize>,
    children: Option<[usize; 2]>,
    items: Vec<usize>,
    height: usize,
    dirty: bool,
}
impl Node {
    fn leaf(parent: Option<usize>) -> Self {
        Self {
            bounds: Aabb::empty(),
            parent,
            children: None,
            items: Vec::new(),
            height: 0,
            dirty: false,
        }
    }
}

/// One native tree. Item IDs retain insertion order; leaf slots do not.
#[derive(Clone)]
struct Tree {
    nodes: Vec<Node>,
    free_nodes: Vec<usize>,
    root: usize,
    bounds: Vec<Aabb>,
    owner: Vec<usize>,
}
impl Tree {
    // Keep BVH insertion indices and refit order explicit.
    #[allow(clippy::needless_range_loop)]
    fn new(bounds: &[Aabb]) -> Self {
        let mut tree = Self {
            nodes: vec![Node::leaf(None)],
            free_nodes: Vec::new(),
            root: 0,
            bounds: bounds.to_vec(),
            owner: vec![0; bounds.len()],
        };
        for id in 0..bounds.len() {
            let node = tree.choose(bounds[id]);
            let refit = tree.add_to_leaf(node, id);
            if refit {
                tree.refit_up(tree.nodes[node].parent, true);
            }
        }
        tree
    }
    fn allocate(&mut self, parent: usize) -> usize {
        if let Some(id) = self.free_nodes.pop() {
            self.nodes[id] = Node::leaf(Some(parent));
            id
        } else {
            let id = self.nodes.len();
            self.nodes.push(Node::leaf(Some(parent)));
            id
        }
    }
    fn add_to_leaf(&mut self, node: usize, id: usize) -> bool {
        let expanded = self.bounds[id].grow(0.5);
        let n = &mut self.nodes[node];
        let refit = n.items.is_empty() || !n.bounds.encloses(expanded);
        if n.items.is_empty() {
            n.bounds = expanded;
        } else if refit {
            n.bounds = n.bounds.merge(expanded);
        }
        n.items.push(id);
        self.owner[id] = node;
        refit
    }
    fn choose(&mut self, bounds: Aabb) -> usize {
        let mut node = self.root;
        loop {
            if let Some([a, b]) = self.nodes[node].children {
                node = if bounds.proximity(self.nodes[a].bounds) < bounds.proximity(self.nodes[b].bounds) {
                    a
                } else {
                    b
                };
            } else if self.nodes[node].items.len() < 128 {
                return node;
            } else {
                return self.split(node, bounds);
            }
        }
    }
    fn partition(a: &mut Vec<usize>, b: &mut Vec<usize>, bounds: &[Aabb], axis: usize, center: f32) {
        let mut i = 0;
        while i < a.len() {
            let p = bounds[a[i]].min;
            if (if axis == 0 { p.x } else { p.y }) > center {
                b.push(a.swap_remove(i));
            } else {
                i += 1;
            }
        }
    }
    fn split(&mut self, node: usize, added: Aabb) -> usize {
        let children = [self.allocate(node), self.allocate(node)];
        let old = std::mem::take(&mut self.nodes[node].items);
        let mut bounds: Vec<Aabb> = old.iter().map(|&i| self.bounds[i]).collect();
        let wildcard = bounds.len();
        bounds.push(added);
        let full = self.nodes[node].bounds;
        let size = full.size();
        let center = size * 0.5 + full.min;
        // Vector2 chooses X for max-axis ties and Y for min-axis ties.
        let order = if size.x < size.y { [1, 0] } else { [0, 1] };
        let coord = |axis| if axis == 0 { center.x } else { center.y };
        let mut a: Vec<usize> = (0..bounds.len()).collect();
        let mut b = Vec::new();
        Self::partition(&mut a, &mut b, &bounds, order[0], coord(order[0]));
        let first_min = a.len().min(b.len());
        if first_min < 32 {
            a.append(&mut b);
            let count = a
                .iter()
                .filter(|&&i| {
                    let p = bounds[i].min;
                    (if order[1] == 0 { p.x } else { p.y }) > coord(order[1])
                })
                .count();
            let second_min = count.min(a.len() - count);
            let best = if second_min > first_min { 1 } else { 0 };
            if first_min.max(second_min) > 0 {
                Self::partition(&mut a, &mut b, &bounds, order[best], coord(order[best]));
            }
        }
        if b.is_empty() {
            b.push(a.swap_remove(0));
        }
        if a.is_empty() {
            a.push(b.swap_remove(0));
        }
        self.nodes[node].children = Some(children);
        let mut target = 0;
        for (child, group) in children.into_iter().zip([a, b]) {
            for index in group {
                if index == wildcard {
                    target = child;
                } else {
                    self.add_to_leaf(child, old[index]);
                }
            }
        }
        self.refit_up(Some(node), false);
        target
    }
    fn refit_node(&mut self, node: usize) {
        let (bounds, height) = if let Some([a, b]) = self.nodes[node].children {
            (
                self.nodes[a].bounds.merge(self.nodes[b].bounds),
                1 + self.nodes[a].height.max(self.nodes[b].height),
            )
        } else {
            (
                self.nodes[node]
                    .items
                    .iter()
                    .fold(Aabb::empty(), |a, &id| a.merge(self.bounds[id]))
                    .grow(0.5),
                0,
            )
        };
        self.nodes[node].bounds = bounds;
        self.nodes[node].height = height;
    }
    fn replace_child(&mut self, parent: Option<usize>, old: usize, new: usize) {
        if let Some(parent) = parent {
            let children = self.nodes[parent].children.as_mut().unwrap();
            if children[0] == old {
                children[0] = new;
            } else {
                children[1] = new;
            }
        } else {
            self.root = new;
        }
        self.nodes[new].parent = parent;
    }
    fn balance(&mut self, a: usize) -> usize {
        let Some([b, c]) = self.nodes[a].children else {
            return a;
        };
        if self.nodes[a].height == 1 {
            return a;
        }
        let delta = self.nodes[c].height as isize - self.nodes[b].height as isize;
        let parent = self.nodes[a].parent;
        if delta > 1 {
            let [f, g] = self.nodes[c].children.unwrap();
            let (keep, moved) = if self.nodes[f].height > self.nodes[g].height {
                (f, g)
            } else {
                (g, f)
            };
            self.replace_child(parent, a, c);
            self.nodes[c].children = Some([a, keep]);
            self.nodes[a].parent = Some(c);
            self.nodes[a].children = Some([b, moved]);
            self.nodes[moved].parent = Some(a);
            self.refit_node(a);
            self.refit_node(c);
            c
        } else if delta < -1 {
            let [d, e] = self.nodes[b].children.unwrap();
            let (keep, moved) = if self.nodes[d].height > self.nodes[e].height {
                (d, e)
            } else {
                (e, d)
            };
            self.replace_child(parent, a, b);
            self.nodes[b].children = Some([keep, a]);
            self.nodes[a].parent = Some(b);
            self.nodes[a].children = Some([moved, c]);
            self.nodes[moved].parent = Some(a);
            self.refit_node(a);
            self.refit_node(b);
            b
        } else {
            a
        }
    }
    fn refit_up(&mut self, mut node: Option<usize>, balance: bool) {
        while let Some(id) = node {
            let id = if balance { self.balance(id) } else { id };
            self.refit_node(id);
            node = self.nodes[id].parent;
        }
    }
    fn remove(&mut self, id: usize) {
        let node = self.owner[id];
        let refit = !self.nodes[node].bounds.grow(-0.5 - 0.001f32).encloses(self.bounds[id]);
        let index = self.nodes[node].items.iter().position(|&i| i == id).unwrap();
        self.nodes[node].items.swap_remove(index);
        if !self.nodes[node].items.is_empty() {
            self.nodes[node].dirty |= refit;
        } else if let Some(parent) = self.nodes[node].parent {
            let children = self.nodes[parent].children.unwrap();
            let sibling = if children[0] == node { children[1] } else { children[0] };
            let grandparent = self.nodes[parent].parent;
            self.replace_child(grandparent, parent, sibling);
            self.free_nodes.push(parent);
            self.free_nodes.push(node);
            self.refit_up(grandparent, false);
        }
    }
    fn refit_dirty(&mut self) {
        let mut stack = vec![self.root];
        while let Some(id) = stack.pop() {
            if let Some([a, b]) = self.nodes[id].children {
                stack.push(a);
                stack.push(b);
            } else if self.nodes[id].dirty {
                self.nodes[id].dirty = false;
                self.refit_up(Some(id), false);
            }
        }
    }
    fn optimize(&mut self, id: usize) {
        self.remove(id);
        let node = self.choose(self.bounds[id]);
        self.add_to_leaf(node, id);
        self.refit_up(Some(node), true);
    }
    fn move_item(&mut self, id: usize, bounds: Aabb) {
        let node = self.owner[id];
        if self.nodes[node].bounds.encloses(bounds) {
            self.bounds[id] = bounds;
            return;
        }
        self.remove(id);
        self.bounds[id] = bounds;
        let node = self.choose(bounds);
        if self.add_to_leaf(node, id) {
            self.refit_up(self.nodes[node].parent, false);
        }
    }
    fn cull(&self, bounds: Aabb, out: &mut Vec<u32>) {
        out.clear();
        let mut stack = vec![self.root];
        while let Some(id) = stack.pop() {
            let node = &self.nodes[id];
            if !bounds.intersects(node.bounds) {
                continue;
            }
            if let Some([a, b]) = node.children {
                stack.push(a);
                stack.push(b);
            } else {
                for &item in &node.items {
                    if bounds.intersects(self.bounds[item]) {
                        out.push(item as u32);
                    }
                }
            }
        }
    }
}

/// A fresh static-wall scene followed by one dynamic vehicle.
/// Geometry-only legacy captures cannot supply the prior native tree state.
#[derive(Clone)]
pub(crate) struct BroadPhase {
    walls: Tree,
    cursor: usize,
    pub(crate) body_node: Aabb,
    pub(crate) initial_pairs: Vec<u32>,
    bodies_first: bool,
}

impl BroadPhase {
    #[cfg(test)]
    pub(crate) fn new(walls: &[Aabb], body: Aabb) -> Self {
        Self::with_layout(walls, body, false)
    }
    pub(crate) fn with_layout(walls: &[Aabb], body: Aabb, tilemap: bool) -> Self {
        let mut tree = Tree::new(walls);
        let mut initial_pairs = Vec::new();
        if tilemap {
            initial_pairs.extend(
                walls
                    .iter()
                    .enumerate()
                    .filter(|(_, wall)| body.intersects(**wall))
                    .map(|(id, _)| id as u32),
            );
        } else {
            tree.cull(body, &mut initial_pairs);
        }
        // Raw PhysicsServer construction adds a shape before assigning space.
        // set_space inserts it immediately, then the pending shape queue applies
        // a second cache update in reverse body order on the first physics tick.
        if !tilemap {
            for (id, &raw) in walls.iter().enumerate().rev() {
                let size = raw.size();
                let margin = ((size.x + size.y) as f64 * 0.5 * 0.05) as f32;
                let d = F2 { x: margin, y: margin };
                let cached = Aabb::from_rect(raw.min - d, size + d * 2.0);
                let density = if initial_pairs.contains(&(id as u32)) {
                    (1.0f64 / 9.0) as f32
                } else {
                    0.0
                };
                tree.move_item(id, cached.grow(0.1 * (1.0 - density)));
            }
        }
        Self {
            walls: tree,
            cursor: 0,
            body_node: body.grow(0.5),
            initial_pairs,
            bodies_first: tilemap,
        }
    }
    pub(crate) fn update(&mut self, body: Aabb) {
        self.walls.refit_dirty();
        if self.cursor > self.walls.bounds.len() {
            self.cursor = 0;
        }
        let id = if self.bodies_first {
            (self.cursor + self.walls.bounds.len()) % (self.walls.bounds.len() + 1)
        } else {
            self.cursor
        };
        if id < self.walls.bounds.len() {
            self.walls.optimize(id);
        } else {
            self.body_node = body.grow(0.5);
        }
        self.cursor += 1;
    }
    pub(crate) fn cull(&self, bounds: Aabb, out: &mut Vec<u32>) {
        self.walls.cull(bounds, out);
    }
    pub(crate) fn overlaps(&self, id: usize, bounds: Aabb) -> bool {
        self.walls.bounds[id].roundtrip_rect().intersects(bounds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generated_tilemap_deferred_wall_query_order() {
        for data in [
            include_str!("../../tests/fixtures/native_tilemap_generated_slot05.json"),
            include_str!("../../tests/fixtures/native_tilemap_generated_slot06.json"),
        ] {
            let data: serde_json::Value = serde_json::from_str(data).unwrap();
            let recipe = &data["recipe"];
            let settings = serde_json::from_str(include_str!("../../assets/random_track_settings.json")).unwrap();
            let template = serde_json::from_str(include_str!("../../assets/scenes/rally_template.json")).unwrap();
            let (_, scene) = crate::track::training_tracks::training_scene_at(
                &template,
                &settings,
                None,
                recipe["seed"].as_i64().unwrap(),
                recipe["generation"].as_u64().unwrap(),
                recipe["slot"].as_u64().unwrap() as usize,
                crate::math::profile::MathProfile::Proton,
            )
            .unwrap();
            let world = crate::track::world::World::from_scene(&scene);
            let walls: Vec<_> = world
                .track
                .shapes
                .iter()
                .map(|shape| Aabb {
                    min: F2::from(shape.min),
                    max: F2::from(shape.max),
                })
                .collect();
            let mut tree = Tree::new(&walls);
            let mut hits = Vec::new();
            let query = Aabb::from_rect(
                F2 {
                    x: -16384.0,
                    y: -16384.0,
                },
                F2 { x: 32768.0, y: 32768.0 },
            );
            for frame in data["frames"].as_array().unwrap() {
                let tick = frame["tick"].as_u64().unwrap() as usize;
                tree.refit_dirty();
                if tick > 3 {
                    tree.optimize(tick - 4);
                }
                tree.cull(query, &mut hits);
                let expected: Vec<_> = frame["walls"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect();
                assert_eq!(hits, expected, "{} walls, native tick {tick}", walls.len());
            }
        }
    }
    #[test]
    fn original_engine_full_wall_query_order() {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/native_broadphase_large80.json")).unwrap();
        let world = crate::track::world::World::from_scene(&data["scene"]);
        let walls: Vec<_> = world
            .track
            .shapes
            .iter()
            .map(|s| Aabb {
                min: F2::from(s.min),
                max: F2::from(s.max),
            })
            .collect();
        let body = Aabb::from_rect(F2 { x: 32.0, y: 65.0 }, F2 { x: 62.0, y: 29.0 });
        let query = Aabb::from_rect(F2 { x: -3000.0, y: -3000.0 }, F2 { x: 4000.0, y: 4000.0 });
        let states = data["states"].as_array().unwrap();
        let expected = |s: &serde_json::Value| -> Vec<u32> {
            s["order"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect()
        };
        let mut tree = BroadPhase::new(&walls, body);
        let mut hits = Vec::new();
        // BroadPhaseOracle waits eight native ticks before setting velocity.
        for _ in 0..8 {
            tree.update(body);
        }
        tree.cull(query, &mut hits);
        assert_eq!(hits, expected(&states[0]), "initial query order");
        for s in states.iter().skip(1) {
            tree.update(body);
            tree.cull(query, &mut hits);
            assert_eq!(hits, expected(s), "query order at tick {}", s["tick"]);
        }
    }
}
