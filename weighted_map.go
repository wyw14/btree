// https://github.com/tidwall/btree
//
// Copyright 2026 Joshua J Baker. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package btree

import (
	"cmp"
	"errors"
	"math"
	"sync/atomic"
	"unsafe"
)

////////////////////////////////////////////////////////////////////////////////
// WeightedMap
////////////////////////////////////////////////////////////////////////////////

// ErrWeightOverflow is returned by WeightedMap.Set when performing the
// operation would cause the total weight of the map to exceed the uint64
// maximum. When this error is returned the map is left completely
// unmodified: no keys, values, weights or tree structures are changed.
var ErrWeightOverflow = errors.New("btree: total weight exceeds uint64 limit")

// WeightedMap is a sorted collection of key value pairs where every item
// carries an explicit uint64 weight supplied by the caller.
//
// Items with a zero weight are fully fledged members of the map: they are
// counted by Len, can be looked up by key, participate in splits and merges,
// but occupy no weighted offsets.
//
// Every node caches the summed weight of its whole subtree and every branch
// slot caches the summed weight of the child subtree, so PrefixWeight and
// SelectWeight only walk a single root-to-leaf path instead of scanning the
// sorted items. All weight sums are guaranteed to never wrap around: a Set
// that would make TotalWeight exceed math.MaxUint64 fails with
// ErrWeightOverflow and changes nothing.
type WeightedMap[K cmp.Ordered, V any] struct {
	cloned bool         // Clone() called at least once.
	count  int          // total number of items, zero weights included
	weight uint64       // cached total weight, equals root.weight
	root   *wnode[K, V] // root node
}

// NewWeightedMap returns a new weighted map.
func NewWeightedMap[K cmp.Ordered, V any]() *WeightedMap[K, V] {
	return new(WeightedMap[K, V])
}

// wnode is a WeightedMap tree node. It intentionally mirrors node but keeps
// its own weight bookkeeping so that the plain Map types pay no per-node
// cost for the weighted feature.
type wnode[K, V any] struct {
	keys    [maxItems]K      // item keys
	values  [maxItems]V      // item values
	weights [maxItems]uint64 // item weights
	rc      atomic.Int64     // copy-on-write reference counter
	len     int              // number of items (key value pairs)
	weight  uint64           // summed weight of the whole subtree
	branch  *wbranch[K, V]   // branch children (nil for leaf)
}

type wbranch[K, V any] struct {
	children [maxItems + 1]*wnode[K, V] // all child nodes
	weights  [maxItems + 1]uint64       // cached subtree weight per child
}

type wbranchAlloc[K, V any] struct {
	node   wnode[K, V]
	branch wbranch[K, V]
}

func (n *wnode[K, V]) leaf() bool {
	return n.branch == nil
}

// recomputeWeight rebuilds the cached subtree weight from the item weights
// and the cached child weights. Every partial sum is bounded by the tree
// total, which can never exceed math.MaxUint64, so no overflow is possible.
func (n *wnode[K, V]) recomputeWeight() uint64 {
	var weight uint64
	for i := 0; i < n.len; i++ {
		weight += n.weights[i]
	}
	if !n.leaf() {
		for i := 0; i <= n.len; i++ {
			weight += n.branch.weights[i]
		}
	}
	n.weight = weight
	return weight
}

// wnodeSearch returns the index of key in n and whether it was found. When
// the key is absent the returned index is the insertion position.
func wnodeSearch[K cmp.Ordered, V any](n *wnode[K, V], key K,
) (i int, found bool) {
	if n.len == 0 {
		return 0, false
	}
	if key > n.keys[n.len-1] {
		return n.len, false
	}
	for key > n.keys[i] {
		i++
	}
	return i, key == n.keys[i]
}

func (b *WeightedMap[K, V]) newNode(leaf bool) *wnode[K, V] {
	var n *wnode[K, V]
	if leaf {
		n = new(wnode[K, V])
	} else {
		a := new(wbranchAlloc[K, V])
		a.node.branch = &a.branch
		n = (*wnode[K, V])(unsafe.Pointer(a))
	}
	return n
}

func (n *wnode[K, V]) insertItemAt(key K, value V, weight uint64, i int) {
	copy(n.keys[i+1:], n.keys[i:n.len])
	n.keys[i] = key
	copy(n.values[i+1:], n.values[i:n.len])
	n.values[i] = value
	copy(n.weights[i+1:], n.weights[i:n.len])
	n.weights[i] = weight
	n.len++
}

func (n *wnode[K, V]) insertChildAt(child *wnode[K, V], i int) {
	// n.len was already incremented by caller.
	copy(n.branch.children[i+1:], n.branch.children[i:n.len])
	n.branch.children[i] = child
	copy(n.branch.weights[i+1:], n.branch.weights[i:n.len])
}

func (b *WeightedMap[K, V]) splitNode(n *wnode[K, V],
) (*wnode[K, V], K, V, uint64) {
	var emptyKey K
	var emptyValue V
	m := maxItems - minItems - 1
	mkey := n.keys[m]
	mvalue := n.values[m]
	mweight := n.weights[m]
	right := b.newNode(n.leaf())
	n.len = m
	right.len = maxItems - m - 1
	copy(right.keys[:], n.keys[m+1:])
	copy(right.values[:], n.values[m+1:])
	copy(right.weights[:], n.weights[m+1:])
	for i := n.len; i < maxItems; i++ {
		n.keys[i] = emptyKey
		n.values[i] = emptyValue
		n.weights[i] = 0
	}
	if !n.leaf() {
		copy(right.branch.children[:], n.branch.children[m+1:])
		copy(right.branch.weights[:], n.branch.weights[m+1:])
		for i := n.len + 1; i <= maxItems; i++ {
			n.branch.children[i] = nil
			n.branch.weights[i] = 0
		}
	}
	n.recomputeWeight()
	right.recomputeWeight()
	return right, mkey, mvalue, mweight
}

//go:noinline
func (b *WeightedMap[K, V]) splitBranch(n *wnode[K, V], i int) {
	right, mkey, mvalue, mweight := b.splitNode(n.branch.children[i])
	n.insertItemAt(mkey, mvalue, mweight, i)
	n.insertChildAt(right, i+1)
	n.branch.weights[i] = n.branch.children[i].weight
	n.branch.weights[i+1] = n.branch.children[i+1].weight
	n.recomputeWeight()
}

// ensure child branch has space to include a new item.
// Return true if split, or false if no change.
func (b *WeightedMap[K, V]) ensureBranch(n *wnode[K, V], i int) bool {
	if n.branch.children[i].len == maxItems {
		b.splitBranch(n, i)
		return true
	}
	return false
}

func (b *WeightedMap[K, V]) splitRoot() {
	root2 := b.newNode(false)
	right, mkey, mvalue, mweight := b.splitNode(b.root)
	root2.branch.children[0] = b.root
	root2.branch.children[1] = right
	root2.insertItemAt(mkey, mvalue, mweight, 0)
	b.root = root2
	root2.branch.weights[0] = root2.branch.children[0].weight
	root2.branch.weights[1] = root2.branch.children[1].weight
	root2.recomputeWeight()
}

// ensure root node has space to include a new item.
func (b *WeightedMap[K, V]) ensureRoot() {
	if b.root.len == maxItems {
		b.splitRoot()
	}
}

func (b *WeightedMap[K, V]) releaseNode(n *wnode[K, V]) {
	if n.rc.Add(-1) < 0 && !n.leaf() {
		for i := 0; i <= n.len; i++ {
			b.releaseNode(n.branch.children[i])
		}
	}
}

func (b *WeightedMap[K, V]) copyNode(n *wnode[K, V]) *wnode[K, V] {
	n2 := b.newNode(n.leaf())
	n2.len = n.len
	n2.weight = n.weight
	copy(n2.keys[:], n.keys[:n.len])
	copy(n2.values[:], n.values[:n.len])
	copy(n2.weights[:], n.weights[:n.len])
	if !n.leaf() {
		for i := 0; i < n.len+1; i++ {
			n.branch.children[i].rc.Add(1)
			n2.branch.children[i] = n.branch.children[i]
		}
		copy(n2.branch.weights[:], n.branch.weights[:n.len+1])
	}
	return n2
}

//go:noinline
func (b *WeightedMap[K, V]) cow0(pn **wnode[K, V]) {
	n2 := b.copyNode(*pn)
	b.releaseNode(*pn)
	*pn = n2
}

// should inline
func (b *WeightedMap[K, V]) cowRoot(mut bool) {
	if mut && b.cloned && b.root != nil && b.root.rc.Load() > 0 {
		b.cow0(&b.root)
	}
}

// should inline
func (b *WeightedMap[K, V]) cowChild(n *wnode[K, V], i int, mut bool) {
	if mut && b.cloned && n.branch.children[i].rc.Load() > 0 {
		b.cow0(&n.branch.children[i])
	}
}

// findWeight performs a read-only lookup, returning the stored weight of key
// and whether the key exists. It never triggers copy-on-write.
func (b *WeightedMap[K, V]) findWeight(key K) (uint64, bool) {
	if b.count == 0 {
		return 0, false
	}
	n := b.root
	for {
		i, found := wnodeSearch(n, key)
		if found {
			return n.weights[i], true
		}
		if n.leaf() {
			return 0, false
		}
		n = n.branch.children[i]
	}
}

// Set inserts or replaces an item and its explicit weight.
// The returned bool is true when an existing item was replaced.
// If the operation would make the total weight exceed math.MaxUint64 the
// call fails with ErrWeightOverflow and the map is left unmodified.
func (b *WeightedMap[K, V]) Set(key K, value V, weight uint64,
) (replaced bool, err error) {
	oldWeight, found := b.findWeight(key)
	if found {
		// New total is b.weight - oldWeight + weight.
		if weight > oldWeight && weight-oldWeight > math.MaxUint64-b.weight {
			return false, ErrWeightOverflow
		}
	} else {
		// New total is b.weight + weight.
		if weight > math.MaxUint64-b.weight {
			return false, ErrWeightOverflow
		}
	}
	if b.count == 0 {
		b.root = b.newNode(true)
		b.root.keys[0] = key
		b.root.values[0] = value
		b.root.weights[0] = weight
		b.root.len = 1
		b.root.weight = weight
		b.count = 1
		b.weight = weight
		return false, nil
	}
	if b.cloned {
		b.cowRoot(true)
	}
	b.ensureRoot()
	replaced = b.nodeSet(b.root, key, value, weight)
	if !replaced {
		b.count++
	}
	b.weight = b.root.weight
	return replaced, nil
}

// nodeSet inserts or replaces an item, maintaining every weight cache on the
// way back up. Only reachable with a non-empty tree.
func (b *WeightedMap[K, V]) nodeSet(n *wnode[K, V], key K, value V,
	weight uint64,
) bool {
	i, found := wnodeSearch(n, key)
	if found {
		oldWeight := n.weights[i]
		n.values[i] = value
		n.weights[i] = weight
		if weight >= oldWeight {
			n.weight += weight - oldWeight
		} else {
			n.weight -= oldWeight - weight
		}
		return true
	}
	if n.leaf() {
		n.insertItemAt(key, value, weight, i)
		n.weight += weight
		return false
	}
	if b.cloned {
		b.cowChild(n, i, true)
	}
	if b.ensureBranch(n, i) {
		return b.nodeSet(n, key, value, weight)
	}
	child := n.branch.children[i]
	replaced := b.nodeSet(child, key, value, weight)
	n.branch.weights[i] = child.weight
	n.recomputeWeight()
	return replaced
}

// nodeDeleteMax removes the maximum item of n and returns it with its
// weight.
func (b *WeightedMap[K, V]) nodeDeleteMax(n *wnode[K, V],
) (K, V, uint64) {
	var emptyKey K
	var emptyValue V
	i := n.len - 1
	if n.leaf() {
		oldKey := n.keys[i]
		oldValue := n.values[i]
		oldWeight := n.weights[i]
		copy(n.keys[i:n.len-1], n.keys[i+1:n.len])
		n.keys[n.len-1] = emptyKey
		copy(n.values[i:n.len-1], n.values[i+1:n.len])
		n.values[n.len-1] = emptyValue
		copy(n.weights[i:n.len-1], n.weights[i+1:n.len])
		n.weights[n.len-1] = 0
		n.len--
		n.weight -= oldWeight
		return oldKey, oldValue, oldWeight
	}
	i++
	b.cowChild(n, i, true)
	oldKey, oldValue, oldWeight := b.nodeDeleteMax(n.branch.children[i])
	// Refresh the child cache before a possible rebalance: rebalance itself
	// rewrites the caches of every slot it touches.
	n.branch.weights[i] = n.branch.children[i].weight
	if n.branch.children[i].len < minItems {
		b.rebalance(n, i)
	}
	n.recomputeWeight()
	return oldKey, oldValue, oldWeight
}

// rebalance the child nodes following a delete operation.
// Provide the index of the child node that has fallen below minItems.
func (b *WeightedMap[K, V]) rebalance(n *wnode[K, V], i int) {
	var emptyKey K
	var emptyValue V
	if i == n.len {
		i--
	}

	// Ensure copy-on-write.
	b.cowChild(n, i, true)
	b.cowChild(n, i+1, true)

	left := n.branch.children[i]
	right := n.branch.children[i+1]

	if left.len+right.len < maxItems {
		// Merge (left,item,right) into the existing left node.
		left.keys[left.len] = n.keys[i]
		copy(left.keys[left.len+1:], right.keys[:right.len])
		left.values[left.len] = n.values[i]
		copy(left.values[left.len+1:], right.values[:right.len])
		left.weights[left.len] = n.weights[i]
		copy(left.weights[left.len+1:], right.weights[:right.len])
		if !left.leaf() {
			copy(left.branch.children[left.len+1:],
				right.branch.children[:right.len+1])
			copy(left.branch.weights[left.len+1:],
				right.branch.weights[:right.len+1])
		}
		left.len += 1 + right.len

		// Move the parent items over one slot.
		copy(n.keys[i:], n.keys[i+1:n.len])
		n.keys[n.len-1] = emptyKey
		copy(n.values[i:], n.values[i+1:n.len])
		n.values[n.len-1] = emptyValue
		copy(n.weights[i:], n.weights[i+1:n.len])
		n.weights[n.len-1] = 0

		// Move the parent children over one slot.
		copy(n.branch.children[i+1:], n.branch.children[i+2:n.len+1])
		n.branch.children[n.len] = nil
		copy(n.branch.weights[i+1:], n.branch.weights[i+2:n.len+1])
		n.branch.weights[n.len] = 0

		n.len--
		left.recomputeWeight()
	} else if left.len > right.len {
		// Move one item left -> right over one slot.
		copy(right.keys[1:], right.keys[:right.len])
		right.keys[0] = n.keys[i]
		n.keys[i] = left.keys[left.len-1]
		left.keys[left.len-1] = emptyKey
		copy(right.values[1:], right.values[:right.len])
		right.values[0] = n.values[i]
		n.values[i] = left.values[left.len-1]
		left.values[left.len-1] = emptyValue
		copy(right.weights[1:], right.weights[:right.len])
		right.weights[0] = n.weights[i]
		n.weights[i] = left.weights[left.len-1]
		left.weights[left.len-1] = 0
		if !left.leaf() {
			copy(right.branch.children[1:],
				right.branch.children[:right.len+1])
			right.branch.children[0] = left.branch.children[left.len]
			left.branch.children[left.len] = nil
			copy(right.branch.weights[1:],
				right.branch.weights[:right.len+1])
			right.branch.weights[0] = left.branch.weights[left.len]
			left.branch.weights[left.len] = 0
		}
		left.len--
		right.len++
		left.recomputeWeight()
		right.recomputeWeight()
	} else {
		// Move one item left <- right over one slot.
		left.keys[left.len] = n.keys[i]
		n.keys[i] = right.keys[0]
		copy(right.keys[:], right.keys[1:right.len])
		right.keys[right.len-1] = emptyKey
		left.values[left.len] = n.values[i]
		n.values[i] = right.values[0]
		copy(right.values[:], right.values[1:right.len])
		right.values[right.len-1] = emptyValue
		left.weights[left.len] = n.weights[i]
		n.weights[i] = right.weights[0]
		copy(right.weights[:], right.weights[1:right.len])
		right.weights[right.len-1] = 0
		if !left.leaf() {
			left.branch.children[left.len+1] = right.branch.children[0]
			copy(right.branch.children[:],
				right.branch.children[1:right.len+1])
			right.branch.children[right.len] = nil
			left.branch.weights[left.len+1] = right.branch.weights[0]
			copy(right.branch.weights[:],
				right.branch.weights[1:right.len+1])
			right.branch.weights[right.len] = 0
		}
		left.len++
		right.len--
		left.recomputeWeight()
		right.recomputeWeight()
	}
	// Recalculate the cached weights for both right and left children.
	// A merge that consumed the last child slot leaves slot i+1 out of the
	// node range (and already zeroed), so it must not be dereferenced.
	n.branch.weights[i] = n.branch.children[i].weight
	if i+1 <= n.len {
		n.branch.weights[i+1] = n.branch.children[i+1].weight
	}
}

func (b *WeightedMap[K, V]) nodeDelete(n *wnode[K, V], key K,
) (V, bool) {
	var emptyValue V
	i, found := wnodeSearch(n, key)
	if n.leaf() {
		if !found {
			return emptyValue, false
		}
		old := n.values[i]
		oldWeight := n.weights[i]
		var emptyKey K
		copy(n.keys[i:n.len-1], n.keys[i+1:n.len])
		n.keys[n.len-1] = emptyKey
		copy(n.values[i:n.len-1], n.values[i+1:n.len])
		n.values[n.len-1] = emptyValue
		copy(n.weights[i:n.len-1], n.weights[i+1:n.len])
		n.weights[n.len-1] = 0
		n.len--
		n.weight -= oldWeight
		return old, true
	}
	var old V
	var deleted bool
	b.cowChild(n, i, true)
	if found {
		old = n.values[i]
		maxKey, maxValue, maxWeight := b.nodeDeleteMax(n.branch.children[i])
		deleted = true
		n.keys[i] = maxKey
		n.values[i] = maxValue
		n.weights[i] = maxWeight
	} else {
		old, deleted = b.nodeDelete(n.branch.children[i], key)
	}
	if !deleted {
		return old, false
	}
	// Refresh the child cache before a possible rebalance: rebalance itself
	// rewrites the caches of every slot it touches.
	n.branch.weights[i] = n.branch.children[i].weight
	if n.branch.children[i].len < minItems {
		b.rebalance(n, i)
	}
	n.recomputeWeight()
	return old, true
}

// Delete removes an item from the map by key.
func (b *WeightedMap[K, V]) Delete(key K) (V, bool) {
	var emptyValue V
	if b.count == 0 {
		return emptyValue, false
	}
	if b.cloned {
		b.cowRoot(true)
	}
	old, deleted := b.nodeDelete(b.root, key)
	if !deleted {
		return old, false
	}
	b.count--
	if b.count == 0 {
		b.root = nil
		b.weight = 0
	} else {
		if b.root.len == 0 && !b.root.leaf() {
			b.root = b.root.branch.children[0]
		}
		b.weight = b.root.weight
	}
	return old, true
}

// Contains returns true if the key exists. Zero-weight items count.
func (b *WeightedMap[K, V]) Contains(key K) bool {
	if b.count == 0 {
		return false
	}
	n := b.root
	for {
		i, found := wnodeSearch(n, key)
		if found {
			return true
		}
		if n.leaf() {
			return false
		}
		n = n.branch.children[i]
	}
}

// Lookup returns the value stored under key. Zero-weight items are
// reachable just like any other item.
func (b *WeightedMap[K, V]) Lookup(key K) (V, bool) {
	var emptyValue V
	if b.count == 0 {
		return emptyValue, false
	}
	n := b.root
	for {
		i, found := wnodeSearch(n, key)
		if found {
			return n.values[i], true
		}
		if n.leaf() {
			return emptyValue, false
		}
		n = n.branch.children[i]
	}
}

// Get returns the value stored under key or the zero value if absent.
func (b *WeightedMap[K, V]) Get(key K) V {
	v, _ := b.Lookup(key)
	return v
}

// Len returns the number of items in the map, including zero-weight items.
func (b *WeightedMap[K, V]) Len() int {
	return b.count
}

// TotalWeight returns the sum of every item weight. It is maintained as a
// cached value and therefore costs O(1).
func (b *WeightedMap[K, V]) TotalWeight() uint64 {
	return b.weight
}

// PrefixWeight returns the summed weight of all items with key strictly
// less than key. It walks a single root-to-leaf path.
func (b *WeightedMap[K, V]) PrefixWeight(key K) uint64 {
	prefix, _ := b.prefixWeight(key, nil)
	return prefix
}

func (b *WeightedMap[K, V]) prefixWeight(key K, visits *int,
) (uint64, int) {
	var v int
	if visits == nil {
		visits = &v
	}
	if b.count == 0 {
		return 0, 0
	}
	var prefix uint64
	n := b.root
	for {
		if visits != nil {
			*visits++
		}
		i, found := wnodeSearch(n, key)
		if n.leaf() {
			for j := 0; j < i; j++ {
				prefix += n.weights[j]
			}
			return prefix, *visits
		}
		// Everything in children[0..i-1] and items[0..i-1] precedes the
		// slot selected by the search.
		for j := 0; j < i; j++ {
			prefix += n.branch.weights[j]
			prefix += n.weights[j]
		}
		if found {
			// All items of child i are strictly less than item i.
			prefix += n.branch.weights[i]
			return prefix, *visits
		}
		n = n.branch.children[i]
	}
}

// prefixWeightCounted is like PrefixWeight but also reports how many nodes
// were visited. It is used by the tests to prove the query follows the tree
// height rather than scanning the sorted items.
func (b *WeightedMap[K, V]) prefixWeightCounted(key K) (uint64, int) {
	var visits int
	return b.prefixWeight(key, &visits)
}

// selectWeightCounted is like SelectWeight but also reports how many nodes
// were visited. It is used by the tests to prove the query follows the tree
// height rather than scanning the sorted items.
func (b *WeightedMap[K, V]) selectWeightCounted(offset uint64,
) (K, V, uint64, bool, int) {
	var visits int
	key, value, itemOffset, ok := b.selectWeight(offset, &visits)
	return key, value, itemOffset, ok, visits
}

// SelectWeight maps a weighted offset to its item. The returned item is the
// unique one for which the prefix weight of all preceding items is <= offset
// and strictly less than the prefix plus the item weight. The returned
// itemOffset is offset minus that prefix weight.
//
// Offsets start at zero. An offset equal to TotalWeight (or larger, or any
// offset when the total weight is zero) never hits and returns ok == false.
// Zero-weight items occupy no offset range and can never be selected.
func (b *WeightedMap[K, V]) SelectWeight(offset uint64,
) (key K, value V, itemOffset uint64, ok bool) {
	return b.selectWeight(offset, nil)
}

func (b *WeightedMap[K, V]) selectWeight(offset uint64, visits *int,
) (K, V, uint64, bool) {
	var emptyKey K
	var emptyValue V
	if b.count == 0 || offset >= b.weight {
		return emptyKey, emptyValue, 0, false
	}
	n := b.root
	for {
		if visits != nil {
			*visits++
		}
		var acc uint64
		if n.leaf() {
			for i := 0; i < n.len; i++ {
				w := n.weights[i]
				if offset < acc+w {
					return n.keys[i], n.values[i], offset - acc, true
				}
				acc += w
			}
			return emptyKey, emptyValue, 0, false
		}
		var next *wnode[K, V]
		for i := 0; i < n.len; i++ {
			// Child i precedes item i in the sorted order.
			cw := n.branch.weights[i]
			if offset < acc+cw {
				next = n.branch.children[i]
				offset -= acc
				break
			}
			acc += cw
			w := n.weights[i]
			if offset < acc+w {
				return n.keys[i], n.values[i], offset - acc, true
			}
			acc += w
		}
		if next == nil {
			// Nothing before the final child contained the offset, so the
			// final child must contain it because offset < b.weight.
			cw := n.branch.weights[n.len]
			if offset < acc+cw {
				next = n.branch.children[n.len]
				offset -= acc
			} else {
				return emptyKey, emptyValue, 0, false
			}
		}
		n = next
	}
}

// Clone the weighted map. This is a fast O(1) operation using
// copy-on-write: later writes to either map never affect the other one,
// including the cached subtree weights, which live inside the nodes.
func (b *WeightedMap[K, V]) Clone() *WeightedMap[K, V] {
	b.cloned = true
	b2 := new(WeightedMap[K, V])
	*b2 = *b
	if b.root != nil {
		b.root.rc.Add(1)
	}
	return b2
}
