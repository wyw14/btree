// https://github.com/tidwall/btree
//
// Copyright 2026 Joshua J Baker. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package btree

import (
	"cmp"
	"iter"
)

////////////////////////////////////////////////////////////////////////////////
// Map diff
////////////////////////////////////////////////////////////////////////////////

// DiffOp describes the kind of change represented by a DiffChange record.
type DiffOp int

const (
	// DiffInsert indicates a key that exists in the new map but not in the
	// old map. Only the DiffChange.New value is meaningful for this operation.
	DiffInsert DiffOp = iota
	// DiffDelete indicates a key that exists in the old map but not in the
	// new map. Only the DiffChange.Old value is meaningful for this operation.
	DiffDelete
	// DiffModify indicates a key that exists in both maps but whose old and
	// new values compare unequal. Both DiffChange.Old and DiffChange.New are
	// meaningful for this operation.
	DiffModify
)

// DiffChange is a single key-level difference yielded by Map.Diff.
//
// Op is the sole indicator of which value fields are meaningful. A stored
// zero value of V is never interpreted as a deletion marker: an absent key is
// always reported through Op (DiffInsert or DiffDelete), so a zero V paired
// with the wrong Op cannot be confused with a missing entry.
type DiffChange[V any] struct {
	Op  DiffOp
	Old V // meaningful for DiffDelete and DiffModify
	New V // meaningful for DiffInsert and DiffModify
}

// Diff returns an iterator over the differences between the receiver, treated
// as the old map, and next, treated as the new map.
//
// Difference records are yielded in ascending key order. Keys present only in
// the receiver are reported as DiffDelete, keys present only in next as
// DiffInsert, and keys present in both are handed to eq; a key is reported as
// DiffModify, carrying both the old and new values, only when eq returns
// false.
//
// V is not required to be comparable: the caller supplies eq to decide value
// equality. Key identity always follows the map's own key ordering; values are
// never compared through reflection, formatting, or anything other than eq.
// The eq function must not be nil.
//
// The walk is strictly read-only: keys and values are only borrowed, the
// configured Copy and Release functions are not called, and node
// copy-on-write reference counts are not changed. Both maps must stay
// unchanged while the iteration is in progress. As with every other iterator
// in this package, concurrent access is not synchronized; callers that share
// maps between goroutines must provide their own coordination and must not
// mutate either input while diffing.
//
// Stopping the iteration early (for example with a break statement) ends the
// walk immediately, and eq is not invoked for any later keys.
//
// Copy-on-write subtrees shared through Clone are exploited directly: when
// both maps reach the same node identity covering the same key range, the
// whole subtree is skipped without descending into it or calling eq. Nodes
// are aligned by key range rather than child index, so a shared subtree is
// still recognized when it sits at a different parent position. Independently
// built maps share no nodes and fall back to an ordered linear merge.
//
// The auxiliary cursors keep at most one frame per tree level, so the extra
// memory is bounded by the heights of the two trees; neither tree is ever
// flattened into a slice.
func (b *Map[K, V]) Diff(next *Map[K, V],
	eq func(old, new V) bool,
) iter.Seq2[K, DiffChange[V]] {
	if eq == nil {
		panic("btree: Map.Diff requires a non-nil value equality function")
	}
	return func(yield func(K, DiffChange[V]) bool) {
		mapDiffRun(&b.tree, &next.tree, eq, yield)
	}
}

// mapDiffRun walks the differences of two trees and returns the number of
// shared copy-on-write subtrees skipped wholesale. It is the single
// implementation backing Map.Diff and is also used by the tests.
func mapDiffRun[K cmp.Ordered, V any](old, next *tree[K, V],
	eq func(V, V) bool, yield func(K, DiffChange[V]) bool,
) (sharedSkips int) {
	// The root is the only node that can never appear as a child piece, so it
	// needs its own identity check. A shared root is an immutable node under
	// copy-on-write rules: equal pointers mean the whole trees are identical.
	if old.root != nil && old.root == next.root {
		return 1
	}
	var w mapDiffWalker[K, V]
	w.eq = eq
	w.cursors[0].init(old)
	w.cursors[1].init(next)
	w.walk(yield)
	return w.sharedSkips
}

// mapDiffWalker holds the two height-bounded cursors and the per-run state.
type mapDiffWalker[K cmp.Ordered, V any] struct {
	cursors     [2]diffCursor[K, V]
	eq          func(old, new V) bool
	sharedSkips int // number of identical shared subtrees skipped
}

// diffCursor presents a tree as an in-order stream of pieces, where a piece
// is either a single item or an entire subtree node. The frame stack never
// grows beyond the height of the tree.
//
// On a branch node the elements alternate in-order:
//
//	child[0], item[0], child[1], item[1], ..., item[len-1], child[len]
//
// encoded so that even element 2*i is child i and odd element 2*i+1 is item i.
// A leaf node exposes its items directly.
type diffCursor[K cmp.Ordered, V any] struct {
	frames []diffFrame[K, V]
}

type diffFrame[K cmp.Ordered, V any] struct {
	n      *node[K, V]
	e      int // current element index within n
	min    K   // cached first key of the child at an even branch element
	minSet bool
}

func (c *diffCursor[K, V]) init(t *tree[K, V]) {
	if t.root != nil {
		c.frames = make([]diffFrame[K, V], 0, t.Height()+1)
		c.frames = append(c.frames, diffFrame[K, V]{n: t.root})
	}
}

func (c *diffCursor[K, V]) done() bool {
	return len(c.frames) == 0
}

// peek returns the current piece. item is true for a single key/value pair and
// false for a subtree, in which case child is the node and key is the first
// key covered by that subtree.
func (c *diffCursor[K, V]) peek() (key K, val V, item bool, child *node[K, V]) {
	f := &c.frames[len(c.frames)-1]
	n := f.n
	if n.leaf() {
		return n.keys[f.e], n.values[f.e], true, nil
	}
	i := f.e >> 1
	if f.e&1 == 1 {
		// Branch separator item.
		return n.keys[i], n.values[i], true, nil
	}
	// Branch child subtree.
	child = n.branch.children[i]
	if !f.minSet {
		f.min = diffNodeMin(child)
		f.minSet = true
	}
	return f.min, val, false, child
}

// descend replaces the current subtree piece with its own first element.
func (c *diffCursor[K, V]) descend() {
	f := &c.frames[len(c.frames)-1]
	child := f.n.branch.children[f.e>>1]
	c.frames = append(c.frames, diffFrame[K, V]{n: child})
}

// advance moves to the next in-order piece, popping exhausted frames.
func (c *diffCursor[K, V]) advance() {
	for len(c.frames) > 0 {
		f := &c.frames[len(c.frames)-1]
		n := f.n
		f.e++
		f.minSet = false
		if n.leaf() {
			if f.e < n.len {
				return
			}
		} else if f.e <= 2*n.len {
			return
		}
		c.frames = c.frames[:len(c.frames)-1]
	}
}

// diffNodeMin returns the smallest key contained in n, following leftmost
// children down to the leaf level.
func diffNodeMin[K cmp.Ordered, V any](n *node[K, V]) K {
	for !n.leaf() {
		n = n.branch.children[0]
	}
	return n.keys[0]
}

// walk performs the ordered structural merge. It never calls a copy-on-write
// entry point and never mutates either tree.
func (w *mapDiffWalker[K, V]) walk(yield func(K, DiffChange[V]) bool) bool {
	a, b := &w.cursors[0], &w.cursors[1]
	for !a.done() || !b.done() {
		if a.done() {
			// Everything left on b is an insertion.
			return w.emitAll(b, DiffInsert, yield)
		}
		if b.done() {
			// Everything left on a is a deletion.
			return w.emitAll(a, DiffDelete, yield)
		}
		ak, av, aItem, aNode := a.peek()
		bk, bv, bItem, bNode := b.peek()
		if ak < bk {
			if aItem {
				if !yield(ak, DiffChange[V]{Op: DiffDelete, Old: av}) {
					return false
				}
				a.advance()
			} else {
				// Refine the subtree until its leading range can be aligned.
				a.descend()
			}
			continue
		}
		if bk < ak {
			if bItem {
				if !yield(bk, DiffChange[V]{Op: DiffInsert, New: bv}) {
					return false
				}
				b.advance()
			} else {
				b.descend()
			}
			continue
		}
		// The two pieces start at the same key.
		switch {
		case aItem && bItem:
			// Same key on both sides; eq is the only value comparison.
			if !w.eq(av, bv) {
				if !yield(ak, DiffChange[V]{
					Op: DiffModify, Old: av, New: bv,
				}) {
					return false
				}
			}
			a.advance()
			b.advance()
		case !aItem && !bItem:
			if aNode == bNode {
				// Same copy-on-write node identity: identical contents over
				// the same key range. Skip the entire subtree without any
				// descent or value comparison.
				w.sharedSkips++
				a.advance()
				b.advance()
			} else {
				// Distinct nodes over overlapping ranges; shapes may differ,
				// so align by key range instead of child index.
				a.descend()
				b.descend()
			}
		case aItem:
			// Let the subtree expose the leaf item matching this key.
			b.descend()
		default:
			a.descend()
		}
	}
	return true
}

// emitAll streams every remaining item of c as records with the given op.
// Subtree pieces are descended through; no value comparisons are needed
// because the other cursor is already exhausted.
func (w *mapDiffWalker[K, V]) emitAll(c *diffCursor[K, V], op DiffOp,
	yield func(K, DiffChange[V]) bool,
) bool {
	for !c.done() {
		key, val, item, _ := c.peek()
		if !item {
			c.descend()
			continue
		}
		ch := DiffChange[V]{Op: op}
		if op == DiffInsert {
			ch.New = val
		} else {
			ch.Old = val
		}
		if !yield(key, ch) {
			return false
		}
		c.advance()
	}
	return true
}
