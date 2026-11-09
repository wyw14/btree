// https://github.com/tidwall/btree
//
// Copyright 2026 Joshua J Baker. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package btree

import (
	"cmp"
	"errors"
	"iter"
)

// ErrMapCursorInvalid is returned by MapCursor operations after the map the
// cursor is bound to has performed a successful write (any operation that
// changed a key or value, including Clear and Release), or after a mutable
// value was handed out through one of the *Mut entry points. Writes performed
// on a different Clone do not invalidate the cursor.
var ErrMapCursorInvalid = errors.New("btree: map cursor invalidated by map mutation")

const cursorMaxInt = int(^uint(0) >> 1)
const cursorMinInt = -cursorMaxInt - 1

// bumpMutSeq wraps a mutable item iterator so that handing out the first
// mutable value invalidates every cursor bound to the same tree. The rule is
// intentionally simple and consistent: once a *Mut iterator yields a value,
// the caller could have changed it, therefore all cursors are invalidated.
// An iterator that yields nothing (empty map, or an out-of-range range) does
// not invalidate anything.
func bumpMutSeq[K, V any](t *tree[K, V], seq iter.Seq2[K, V]) iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		var bumped bool
		seq(func(key K, value V) bool {
			if !bumped {
				t.ver++
				bumped = true
			}
			return yield(key, value)
		})
	}
}

// mapCursorFrame is one level of the cursor's saved root-to-leaf node path.
// For internal levels the index i is the child slot used to descend.
// For the level that owns the current item (see MapCursor.lev), i is the
// index of the current item inside that node, which may be an internal node
// when the current item is a branch separator.
type mapCursorFrame[K, V any] struct {
	n   *node[K, V]
	i   int
	off int // global ordinal of this node's first item (seek bookkeeping)
}

// MapCursor is a bidirectional cursor over a Map.
//
// A cursor has three kinds of position: a valid item position (Index in
// [0,Len)), the before-first sentinel (logical index -1) and the after-last
// sentinel (logical index Len). A freshly created cursor starts at
// before-first.
//
// A cursor is bound to the structure version of its map at creation time.
// Any successful write on that map afterwards (changing an item, Clear or
// Release), or obtaining a mutable value through a *Mut entry point,
// invalidates the cursor and every subsequent call returns
// ErrMapCursorInvalid. Writes on a separate Clone do not invalidate it.
// Operations that change nothing (missed Replace/Delete, out-of-range
// deletes, ...) do not invalidate it.
//
// A cursor is read-only: it never invokes the map's Copy/Release hooks and
// does not own any part of the tree. Mutating a value through an external
// pointer is not detected.
//
// MapCursor is intended for single-goroutine use. It is not safe to use a
// cursor, or the bound map, from multiple goroutines concurrently.
type MapCursor[K cmp.Ordered, V any] struct {
	b   *Map[K, V]
	ver uint64 // version bound at creation
	n   int    // number of items at bind time

	// frames is the full root-to-leaf path, allocated once at creation.
	// Tree height cannot change while the cursor stays valid.
	frames []mapCursorFrame[K, V]
	lev    int // level owning the current item (0..len(frames)-1)
	pos    int // -1 before-first, 0..n-1 valid, n after-last
	err    error

	// steps counts primitive node/item steps since cursor creation. It exists
	// purely to let tests verify amortized-constant stepping (no hidden
	// re-walks from the root). It is not part of the public semantics.
	steps int
}

// Cursor returns a new bidirectional cursor bound to the current structure
// version of the map. The cursor starts on the before-first sentinel.
func (b *Map[K, V]) Cursor() *MapCursor[K, V] {
	b.init()
	c := &MapCursor[K, V]{
		b:   b,
		ver: b.tree.ver,
		n:   b.tree.count,
		pos: -1,
	}
	if b.tree.count > 0 {
		// Preallocate the whole path now so that stepping never allocates.
		depth := 1
		for n := b.tree.root; !n.leaf(); n = n.branch.children[0] {
			depth++
		}
		c.frames = make([]mapCursorFrame[K, V], depth)
	}
	return c
}

// invalid checks the bound version and latches the invalidation error.
func (c *MapCursor[K, V]) invalid() bool {
	if c.err != nil {
		return true
	}
	if c.b.tree.ver != c.ver {
		c.err = ErrMapCursorInvalid
		return true
	}
	return false
}

// Err returns nil while the cursor is valid and the stable
// ErrMapCursorInvalid error once the bound map has changed.
func (c *MapCursor[K, V]) Err() error {
	if c.invalid() {
		return c.err
	}
	return nil
}

// Valid reports whether the cursor currently sits on an item. It returns
// false at either sentinel and after invalidation.
func (c *MapCursor[K, V]) Valid() bool {
	return !c.invalid() && c.pos >= 0 && c.pos < c.n
}

// BeforeFirst reports whether the cursor sits on the before-first sentinel.
func (c *MapCursor[K, V]) BeforeFirst() bool {
	return !c.invalid() && c.pos == -1
}

// AfterLast reports whether the cursor sits on the after-last sentinel.
func (c *MapCursor[K, V]) AfterLast() bool {
	return !c.invalid() && c.pos == c.n
}

// Item returns the current key and value. The boolean is false on the
// before-first/after-last sentinels and after invalidation; in those cases
// the returned key and value are zero values and must not be interpreted as
// a stored item, even when zero is a legitimate stored key or value.
func (c *MapCursor[K, V]) Item() (K, V, bool) {
	var key K
	var value V
	if c.invalid() || c.pos < 0 || c.pos >= c.n {
		return key, value, false
	}
	f := &c.frames[c.lev]
	return f.n.keys[f.i], f.n.values[f.i], true
}

// Key returns the current key. See Item for the boolean semantics.
func (c *MapCursor[K, V]) Key() (K, bool) {
	key, _, ok := c.Item()
	return key, ok
}

// Value returns the current value. See Item for the boolean semantics.
func (c *MapCursor[K, V]) Value() (V, bool) {
	_, value, ok := c.Item()
	return value, ok
}

// Index returns the zero-based ordinal of the current item. It returns false
// on either sentinel and after invalidation.
func (c *MapCursor[K, V]) Index() (int, bool) {
	if c.invalid() || c.pos < 0 || c.pos >= c.n {
		return 0, false
	}
	return c.pos, true
}

// First positions the cursor on the first item.
// On an empty map it lands on the after-last sentinel and returns false.
func (c *MapCursor[K, V]) First() (bool, error) {
	if c.invalid() {
		return false, c.err
	}
	if c.n == 0 {
		c.pos = 0
		return false, nil
	}
	c.gotoIndex(0)
	return true, nil
}

// Last positions the cursor on the last item.
// On an empty map it lands on the before-first sentinel and returns false.
func (c *MapCursor[K, V]) Last() (bool, error) {
	if c.invalid() {
		return false, c.err
	}
	if c.n == 0 {
		c.pos = -1
		return false, nil
	}
	c.gotoIndex(c.n - 1)
	return true, nil
}

// SeekGE positions the cursor on the first item whose key is greater than or
// equal to key. If no such item exists it lands on the after-last sentinel.
func (c *MapCursor[K, V]) SeekGE(key K) (bool, error) {
	if c.invalid() {
		return false, c.err
	}
	if c.n == 0 {
		c.pos = 0
		return false, nil
	}
	return c.seekGE(key)
}

// SeekIndex positions the cursor on the item at index (0-based). Negative
// indexes land on the before-first sentinel and indexes greater than or
// equal to Len land on the after-last sentinel.
func (c *MapCursor[K, V]) SeekIndex(index int) (bool, error) {
	if c.invalid() {
		return false, c.err
	}
	switch {
	case index < 0:
		c.pos = -1
		return false, nil
	case index >= c.n:
		c.pos = c.n
		return false, nil
	default:
		c.gotoIndex(index)
		return true, nil
	}
}

// Next advances to the next item. From the before-first sentinel it enters
// the first item; from the last item it lands on after-last; on after-last
// it stays there. On an empty map, Next from before-first lands on
// after-last.
func (c *MapCursor[K, V]) Next() (bool, error) {
	if c.invalid() {
		return false, c.err
	}
	return c.next(), nil
}

// Prev moves to the previous item. From the after-last sentinel it enters
// the last item; from the first item it lands on before-first; on
// before-first it stays there. On an empty map, Prev from after-last lands
// on before-first.
func (c *MapCursor[K, V]) Prev() (bool, error) {
	if c.invalid() {
		return false, c.err
	}
	return c.prev(), nil
}

// Skip moves relative to the current position: Skip(1) is equivalent to
// Next, Skip(-1) to Prev. The sentinels participate with logical indexes -1
// (before-first) and Len (after-last). A jump that would leave the valid
// range stops exactly on the corresponding sentinel, regardless of the
// magnitude of delta and without integer overflow. Skip(0) reports whether
// the cursor is on an item without moving.
func (c *MapCursor[K, V]) Skip(delta int) (bool, error) {
	if c.invalid() {
		return false, c.err
	}
	switch {
	case delta == 0:
		return c.pos >= 0 && c.pos < c.n, nil
	case delta > 0:
		if c.pos < 0 {
			// From before-first (-1) the after-last sentinel is reached
			// exactly when delta > n. Compare without adding n+1 to avoid
			// overflow at the max-int capacity.
			if delta == cursorMaxInt || delta > c.n {
				c.pos = c.n
				return false, nil
			}
			c.gotoIndex(c.pos + delta)
			return true, nil
		}
		// Distance from pos to the after-last sentinel is c.n-c.pos.
		if delta >= c.n-c.pos {
			c.pos = c.n
			return false, nil
		}
		c.gotoIndex(c.pos + delta)
		return true, nil
	default:
		// Guard -delta against the most-negative int before negating.
		if delta == cursorMinInt || -delta >= c.pos+1 {
			c.pos = -1
			return false, nil
		}
		c.gotoIndex(c.pos + delta)
		return true, nil
	}
}

// search picks the right per-node search (string prefix, binary or linear)
// exactly like the map's own operations.
func (c *MapCursor[K, V]) search(n *node[K, V], key K) (int, bool) {
	var zero V
	if c.b.tree.strpre {
		return c.b.strSearch(n, prefixString(key), key)
	}
	return c.b.tree.search(n, key, zero)
}

// gotoIndex rebuilds the root-to-item path from branch counts in O(height).
func (c *MapCursor[K, V]) gotoIndex(index int) {
	r := index
	d := 0
	n := c.b.tree.root
	for {
		c.frames[d].n = n
		if n.leaf() {
			c.frames[d].i = r
			c.lev = d
			c.pos = index
			return
		}
		i := 0
		for ; i < n.len; i++ {
			cnt := n.branch.counts[i]
			if r < cnt {
				break
			}
			if r == cnt {
				// Current item is separator i of this internal node.
				c.frames[d].i = i
				c.lev = d
				c.pos = index
				return
			}
			r -= cnt + 1
		}
		c.frames[d].i = i
		d++
		n = n.branch.children[i]
		c.steps++
	}
}

// seekGE descends along the search path for key using branch counts to track
// the global ordinal. O(height).
func (c *MapCursor[K, V]) seekGE(key K) (bool, error) {
	base := 0
	d := 0
	n := c.b.tree.root
	for {
		c.frames[d].n = n
		c.frames[d].off = base
		i, found := c.search(n, key)
		if found {
			rank := base + i
			if !n.leaf() {
				for j := 0; j <= i; j++ {
					rank += n.branch.counts[j]
				}
			}
			c.frames[d].i = i
			c.lev = d
			c.pos = rank
			return true, nil
		}
		if n.leaf() {
			if i < n.len {
				c.frames[d].i = i
				c.lev = d
				c.pos = base + i
				return true, nil
			}
			// The whole descended subtree is smaller than key: climb the
			// saved path and land on the first parent separator that is
			// still in play, or fall off the right edge at the root.
			for d > 0 {
				c.steps++
				d--
				parent := c.frames[d].n
				s := c.frames[d].i // slot of the exhausted child
				if s < parent.len {
					rank := c.frames[d].off + s
					for j := 0; j <= s; j++ {
						rank += parent.branch.counts[j]
					}
					c.frames[d].i = s
					c.lev = d
					c.pos = rank
					return true, nil
				}
			}
			c.pos = c.n
			return false, nil
		}
		c.frames[d].i = i
		base += i
		for j := 0; j < i; j++ {
			base += n.branch.counts[j]
		}
		d++
		n = n.branch.children[i]
		c.steps++
	}
}

// descend fills the path below internal level d by entering child slot and
// repeatedly taking the leftmost (left == true) or rightmost child until a
// leaf is reached.
func (c *MapCursor[K, V]) descend(d, slot int, left bool) {
	n := c.frames[d].n
	c.frames[d].i = slot
	n = n.branch.children[slot]
	c.steps++
	for d++; ; d++ {
		c.frames[d].n = n
		if n.leaf() {
			if left {
				c.frames[d].i = 0
			} else {
				c.frames[d].i = n.len - 1
			}
			c.lev = d
			return
		}
		i := 0
		if !left {
			i = n.len
		}
		c.frames[d].i = i
		n = n.branch.children[i]
		c.steps++
	}
}

// next performs one local step, amortized O(1) node work.
func (c *MapCursor[K, V]) next() bool {
	switch {
	case c.pos >= c.n:
		// Already on after-last (also covers the empty map).
		return false
	case c.pos < 0:
		// before-first: enter the first item.
		if c.n == 0 {
			c.pos = 0
			return false
		}
		c.gotoIndex(0)
		return true
	}
	d := c.lev
	f := &c.frames[d]
	if d < len(c.frames)-1 {
		// The current item is an internal separator f.i: the next item is
		// the leftmost item of child f.i+1.
		c.descend(d, f.i+1, true)
		c.pos++
		return true
	}
	// Current item lives in the leaf.
	if f.i+1 < f.n.len {
		f.i++
		c.steps++
		c.pos++
		return true
	}
	// Leaf exhausted: walk up until a parent offers its separator, or fall
	// off the right edge.
	for d > 0 {
		c.steps++
		d--
		ci := c.frames[d].i // slot of the exhausted subtree in its parent
		if ci < c.frames[d].n.len {
			c.frames[d].i = ci
			c.lev = d
			c.pos++
			return true
		}
	}
	c.pos = c.n
	return false
}

// prev performs one local step, amortized O(1) node work.
func (c *MapCursor[K, V]) prev() bool {
	switch {
	case c.pos < 0:
		// Already on before-first (also covers the empty map).
		return false
	case c.pos >= c.n:
		// after-last: enter the last item.
		if c.n == 0 {
			c.pos = -1
			return false
		}
		c.gotoIndex(c.n - 1)
		return true
	}
	d := c.lev
	f := &c.frames[d]
	if d < len(c.frames)-1 {
		// The current item is an internal separator f.i: the previous item
		// is the rightmost item of child f.i.
		c.descend(d, f.i, false)
		c.pos--
		return true
	}
	// Current item lives in the leaf.
	if f.i > 0 {
		f.i--
		c.steps++
		c.pos--
		return true
	}
	// At the leaf's first item: walk up until a parent separator exists, or
	// fall off the left edge.
	for d > 0 {
		c.steps++
		d--
		ci := c.frames[d].i // slot of this subtree in its parent
		if ci > 0 {
			c.frames[d].i = ci - 1
			c.lev = d
			c.pos--
			return true
		}
	}
	c.pos = -1
	return false
}
