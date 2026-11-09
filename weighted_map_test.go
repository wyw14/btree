// https://github.com/tidwall/btree
//
// Copyright 2026 Joshua J Baker. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package btree

import (
	"cmp"
	"fmt"
	"maps"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

////////////////////////////////////////////////////////////////////////////////
// helpers
////////////////////////////////////////////////////////////////////////////////

type wmEntry[K cmp.Ordered, V any] struct {
	k K
	v V
	w uint64
}

type wmItem[V any] struct {
	v V
	w uint64
}

func wmMinKey[K cmp.Ordered, V any](n *wnode[K, V]) K {
	for !n.leaf() {
		n = n.branch.children[0]
	}
	return n.keys[0]
}

func wmMaxKey[K cmp.Ordered, V any](n *wnode[K, V]) K {
	for !n.leaf() {
		n = n.branch.children[n.len]
	}
	return n.keys[n.len-1]
}

// wmCheckNode walks the whole tree and verifies every cached weight, the
// child weight caches, the key ordering and the B-tree node sizes. It
// returns the number of items found.
func wmCheckNode[K cmp.Ordered, V any](n *wnode[K, V], isRoot bool) int {
	assert(n != nil)
	assert(n.len >= 1)
	assert(n.len <= maxItems)
	if !isRoot {
		assert(n.len >= minItems)
	}
	for i := 1; i < n.len; i++ {
		assert(n.keys[i-1] < n.keys[i])
	}
	var sum uint64
	for i := 0; i < n.len; i++ {
		sum += n.weights[i]
	}
	count := n.len
	if n.leaf() {
		assert(n.branch == nil)
	} else {
		for i := 0; i <= n.len; i++ {
			ch := n.branch.children[i]
			assert(ch != nil)
			// Child slot cache must match the child subtree weight.
			assert(n.branch.weights[i] == ch.weight)
			if i < n.len {
				assert(wmMaxKey(ch) < n.keys[i])
			}
			if i > 0 {
				assert(wmMinKey(ch) > n.keys[i-1])
			}
			sum += n.branch.weights[i]
			count += wmCheckNode(ch, false)
		}
	}
	// Node cache must equal the freshly summed weight.
	assert(sum == n.weight)
	return count
}

// wmCheckInvariants validates the public counters and every internal weight
// invariant.
func wmCheckInvariants[K cmp.Ordered, V any](b *WeightedMap[K, V]) {
	if b.count == 0 {
		assert(b.root == nil)
		assert(b.weight == 0)
		assert(b.TotalWeight() == 0)
		assert(b.Len() == 0)
		return
	}
	assert(b.root != nil)
	count := wmCheckNode(b.root, true)
	assert(count == b.count)
	assert(count == b.Len())
	assert(b.root.weight == b.weight)
	assert(b.TotalWeight() == b.weight)
}

func wmHeight[K cmp.Ordered, V any](b *WeightedMap[K, V]) int {
	if b.root == nil {
		return 0
	}
	h := 0
	n := b.root
	for !n.leaf() {
		n = n.branch.children[0]
		h++
	}
	return h
}

// wmCollect returns all items in sorted order, including stored weights.
func wmCollect[K cmp.Ordered, V any](b *WeightedMap[K, V],
) []wmEntry[K, V] {
	var out []wmEntry[K, V]
	var walk func(n *wnode[K, V])
	walk = func(n *wnode[K, V]) {
		if n.leaf() {
			for i := 0; i < n.len; i++ {
				out = append(out, wmEntry[K, V]{
					n.keys[i], n.values[i], n.weights[i],
				})
			}
			return
		}
		for i := 0; i < n.len; i++ {
			walk(n.branch.children[i])
			out = append(out, wmEntry[K, V]{
				n.keys[i], n.values[i], n.weights[i],
			})
		}
		walk(n.branch.children[n.len])
	}
	if b.root != nil {
		walk(b.root)
	}
	return out
}

func naiveTotal(model map[int]wmItem[int]) uint64 {
	var total uint64
	for _, it := range model {
		total += it.w
	}
	return total
}

func naivePrefix(model map[int]wmItem[int], sorted []int, key int) uint64 {
	var sum uint64
	for _, k := range sorted {
		if k >= key {
			break
		}
		sum += model[k].w
	}
	return sum
}

func naiveSelect(model map[int]wmItem[int], sorted []int, offset uint64,
) (key int, itemOffset uint64, ok bool) {
	var acc uint64
	for _, k := range sorted {
		w := model[k].w
		if offset < acc+w {
			return k, offset - acc, true
		}
		acc += w
	}
	return 0, 0, false
}

func wmExpectSelect(b *WeightedMap[int, int], expKey, expValue int,
	expItemOffset, offset uint64,
) {
	k, v, itemOffset, ok := b.SelectWeight(offset)
	assert(ok)
	assert(k == expKey)
	assert(v == expValue)
	assert(itemOffset == expItemOffset)
}

////////////////////////////////////////////////////////////////////////////////
// tests
////////////////////////////////////////////////////////////////////////////////

func TestWeightedMapEmpty(t *testing.T) {
	b := NewWeightedMap[int, int]()
	wmCheckInvariants(b)
	assert(b.Len() == 0)
	assert(b.TotalWeight() == 0)
	assert(b.PrefixWeight(42) == 0)
	_, _, _, ok := b.SelectWeight(0)
	assert(!ok)
	_, ok = b.Lookup(42)
	assert(!ok)
	assert(!b.Contains(42))
	_, ok = b.Delete(42)
	assert(!ok)
	assert(b.Get(42) == 0)
}

func TestWeightedMapBasic(t *testing.T) {
	b := NewWeightedMap[int, string]()
	replaced, err := b.Set(10, "ten", 3)
	assert(err == nil && !replaced)
	replaced, err = b.Set(20, "twenty", 5)
	assert(err == nil && !replaced)
	replaced, err = b.Set(30, "thirty", 7)
	assert(err == nil && !replaced)
	assert(b.Len() == 3)
	assert(b.TotalWeight() == 15)

	// Replace with a different weight.
	replaced, err = b.Set(20, "twenty2", 2)
	assert(err == nil && replaced)
	assert(b.Len() == 3)
	assert(b.TotalWeight() == 12)
	assert(b.Get(20) == "twenty2")

	// Prefix sums: strictly less than key.
	assert(b.PrefixWeight(10) == 0)
	assert(b.PrefixWeight(20) == 3)
	assert(b.PrefixWeight(30) == 5)
	assert(b.PrefixWeight(25) == 5)
	assert(b.PrefixWeight(100) == 12)
	assert(b.PrefixWeight(0) == 0)

	// Select every legal offset.
	expected := []struct {
		key    int
		weight uint64
		value  string
	}{
		{10, 3, "ten"},
		{20, 2, "twenty2"},
		{30, 7, "thirty"},
	}
	var off uint64
	for _, e := range expected {
		for d := uint64(0); d < e.weight; d++ {
			k, v, itemOffset, ok := b.SelectWeight(off + d)
			assert(ok)
			assert(k == e.key)
			assert(v == e.value)
			assert(itemOffset == d)
		}
		off += e.weight
	}
	// Offset equal to the total never hits.
	_, _, _, ok := b.SelectWeight(12)
	assert(!ok)

	// Delete the middle item.
	v, deleted := b.Delete(20)
	assert(deleted && v == "twenty2")
	assert(b.Len() == 2)
	assert(b.TotalWeight() == 10)
	assert(b.PrefixWeight(30) == 3)
	wmCheckInvariants(b)
}

func TestWeightedMapZeroWeights(t *testing.T) {
	b := NewWeightedMap[int, string]()
	const n = 5000
	model := make(map[int]uint64)
	for i := 0; i < n; i++ {
		// Roughly one third of the items carry no weight, sandwiched in
		// between weighted items.
		var w uint64
		if i%3 != 0 {
			w = uint64(i%5) + 1
		}
		_, err := b.Set(i, fmt.Sprintf("v%d", i), w)
		assert(err == nil)
		model[i] = w
	}
	wmCheckInvariants(b)
	assert(b.Len() == n)

	// Zero-weight items are still present and reachable by key.
	zeroCount := 0
	var expTotal uint64
	for i := 0; i < n; i++ {
		assert(b.Contains(i))
		assert(b.Get(i) == fmt.Sprintf("v%d", i))
		if model[i] == 0 {
			zeroCount++
		} else {
			expTotal += model[i]
		}
	}
	assert(zeroCount > 0)
	assert(b.TotalWeight() == expTotal)

	// Every legal offset must resolve to a weighted item, never a
	// zero-weight one, and must match the naive model.
	sorted := slices.Sorted(maps.Keys(model))
	for off := uint64(0); off < expTotal; off++ {
		k, _, itemOffset, ok := b.SelectWeight(off)
		assert(ok)
		assert(model[k] > 0)
		ek, eio, eok := naiveSelectWeighted(model, sorted, off)
		assert(eok)
		assert(k == ek && itemOffset == eio)
	}
	// The total offset and beyond never hit.
	_, _, _, ok := b.SelectWeight(expTotal)
	assert(!ok)
	_, _, _, ok = b.SelectWeight(expTotal + 1)
	assert(!ok)

	// PrefixWeight agrees with the naive model for every key and a set of
	// absent keys.
	for i := 0; i < n; i++ {
		assert(b.PrefixWeight(i) == naivePrefixWeighted(model, sorted, i))
	}
	for _, key := range []int{-10, -1, n, n + 10, n/2*2 + 1} {
		assert(b.PrefixWeight(key) == naivePrefixWeighted(model, sorted, key))
	}

	// Deleting a zero-weight item lowers Len but not TotalWeight.
	_, deleted := b.Delete(0)
	assert(deleted)
	delete(model, 0)
	assert(b.Len() == n-1)
	assert(b.TotalWeight() == expTotal)
	wmCheckInvariants(b)

	// Deleting every remaining item empties the tree without any weight
	// accounting error.
	for i := 1; i < n; i++ {
		_, deleted := b.Delete(i)
		assert(deleted)
	}
	wmCheckInvariants(b)
	assert(b.Len() == 0)
	assert(b.TotalWeight() == 0)
}

// naiveSelectWeighted mirrors SelectWeight on a key->weight model.
func naiveSelectWeighted(model map[int]uint64, sorted []int, offset uint64,
) (key int, itemOffset uint64, ok bool) {
	var acc uint64
	for _, k := range sorted {
		w := model[k]
		if offset < acc+w {
			return k, offset - acc, true
		}
		acc += w
	}
	return 0, 0, false
}

// naivePrefixWeighted mirrors PrefixWeight on a key->weight model.
func naivePrefixWeighted(model map[int]uint64, sorted []int, key int) uint64 {
	var sum uint64
	for _, k := range sorted {
		if k >= key {
			break
		}
		sum += model[k]
	}
	return sum
}

func TestWeightedMapSelectBoundaries(t *testing.T) {
	// A multi-level tree where every offset boundary, including offsets
	// sitting exactly on branch children and separator items, is checked
	// against the naive model.
	b := NewWeightedMap[int, int]()
	model := make(map[int]uint64)
	const n = 8000
	rng := rand.New(rand.NewPCG(7, 13))
	for i := 0; i < n; i++ {
		w := uint64(rng.IntN(6)) // 0..5, plenty of zeros and boundaries
		_, err := b.Set(i, i*3+1, w)
		assert(err == nil)
		model[i] = w
	}
	wmCheckInvariants(b)
	assert(wmHeight(b) >= 2)
	sorted := slices.Sorted(maps.Keys(model))
	var expTotal uint64
	for _, w := range model {
		expTotal += w
	}

	// Scan every single legal offset.
	for off := uint64(0); off < expTotal; off++ {
		k, v, itemOffset, ok := b.SelectWeight(off)
		assert(ok)
		ek, eio, eok := naiveSelectWeighted(model, sorted, off)
		assert(eok)
		assert(k == ek)
		assert(v == ek*3+1)
		assert(itemOffset == eio)
	}
	_, _, _, ok := b.SelectWeight(expTotal)
	assert(!ok)

	// Explicitly probe the first and last offset of every weighted item.
	var acc uint64
	for _, k := range sorted {
		w := model[k]
		if w > 0 {
			wmExpectSelect(b, k, k*3+1, 0, acc)
			wmExpectSelect(b, k, k*3+1, w-1, acc+w-1)
		}
		acc += w
	}
	assert(acc == expTotal)
}

func TestWeightedMapMaxUint64(t *testing.T) {
	b := NewWeightedMap[int, int]()
	max := uint64(math.MaxUint64)

	_, err := b.Set(1, 10, max)
	assert(err == nil)
	assert(b.TotalWeight() == max)
	assert(b.Len() == 1)

	// The single item owns the whole offset range.
	k, v, itemOffset, ok := b.SelectWeight(0)
	assert(ok && k == 1 && v == 10 && itemOffset == 0)
	k, _, itemOffset, ok = b.SelectWeight(max - 1)
	assert(ok && k == 1 && itemOffset == max-1)
	_, _, _, ok = b.SelectWeight(max)
	assert(!ok)

	assert(b.PrefixWeight(1) == 0)
	assert(b.PrefixWeight(2) == max)

	// A zero-weight item is allowed even when the total is already maxed.
	replaced, err := b.Set(2, 20, 0)
	assert(err == nil && !replaced)
	assert(b.Len() == 2)
	assert(b.TotalWeight() == max)
	assert(b.Contains(2))

	// Any positive weight now overflows and changes nothing.
	_, err = b.Set(3, 30, 1)
	assert(err == ErrWeightOverflow)
	assert(!b.Contains(3))
	assert(b.Len() == 2)
	assert(b.TotalWeight() == max)
	wmCheckInvariants(b)

	// Replacing with the same maximum weight is fine.
	replaced, err = b.Set(1, 11, max)
	assert(err == nil && replaced)
	assert(b.Get(1) == 11)
	assert(b.TotalWeight() == max)

	// Reduce the weight, then fill the tree back up to exactly the limit.
	replaced, err = b.Set(1, 11, max-1)
	assert(err == nil && replaced)
	assert(b.TotalWeight() == max-1)
	_, err = b.Set(3, 30, 1)
	assert(err == nil)
	assert(b.TotalWeight() == max)
	_, err = b.Set(4, 40, 1)
	assert(err == ErrWeightOverflow)
	assert(!b.Contains(4))
	assert(b.TotalWeight() == max)

	// Deleting frees weight room again.
	_, deleted := b.Delete(1)
	assert(deleted)
	assert(b.TotalWeight() == 1)
	wmCheckInvariants(b)
}

func TestWeightedMapOverflowUnchanged(t *testing.T) {
	b := NewWeightedMap[int, int]()
	for i, w := range []uint64{10, 20, 30, 0, 40} {
		_, err := b.Set(i+1, (i+1)*100, w)
		assert(err == nil)
	}
	before := wmCollect(b)
	beforeTotal := b.TotalWeight()
	beforeLen := b.Len()

	// A replacement that would overflow must leave everything observable
	// untouched: keys, values, weights and the tree structure.
	_, err := b.Set(2, 999, math.MaxUint64)
	assert(err == ErrWeightOverflow)
	assert(wmCollect(b) == nil || len(wmCollect(b)) == len(before))
	after := wmCollect(b)
	assert(len(after) == len(before))
	for i := range before {
		assert(after[i] == before[i])
	}
	assert(b.TotalWeight() == beforeTotal)
	assert(b.Len() == beforeLen)
	wmCheckInvariants(b)

	// Same for an inserting overflow.
	_, err = b.Set(99, 999, math.MaxUint64)
	assert(err == ErrWeightOverflow)
	after = wmCollect(b)
	assert(len(after) == len(before))
	for i := range before {
		assert(after[i] == before[i])
	}
	assert(!b.Contains(99))
	wmCheckInvariants(b)

	// A replacement that brings the total to exactly math.MaxUint64 is
	// accepted: newTotal = beforeTotal - 20 + legal == MaxUint64.
	legal := math.MaxUint64 - (beforeTotal - 20)
	replaced, err := b.Set(2, 222, legal)
	assert(err == nil && replaced)
	assert(b.TotalWeight() == math.MaxUint64)
	assert(b.Get(2) == 222)
	wmCheckInvariants(b)

	// One more unit of weight anywhere must now overflow, even inserting a
	// brand new key.
	_, err = b.Set(6, 600, 1)
	assert(err == ErrWeightOverflow)
	assert(!b.Contains(6))
	// An existing key is completely untouched.
	assert(b.Get(5) == 500)
	wmCheckInvariants(b)
}

func TestWeightedMapRootSplitShrink(t *testing.T) {
	b := NewWeightedMap[int, int]()
	const n = 20000
	for i := 0; i < n; i++ {
		_, err := b.Set(i, i, uint64(i%7))
		assert(err == nil)
		if (i+1)%1000 == 0 {
			wmCheckInvariants(b)
		}
	}
	wmCheckInvariants(b)
	assert(b.Len() == n)
	assert(wmHeight(b) >= 2)

	// Delete the even keys, forcing borrows and merges.
	for i := 0; i < n; i += 2 {
		_, ok := b.Delete(i)
		assert(ok)
	}
	wmCheckInvariants(b)
	assert(b.Len() == n/2)
	for i := 1; i < n; i += 2 {
		assert(b.Contains(i))
	}

	// Remove the rest and watch the root collapse.
	for i := 1; i < n; i += 2 {
		_, ok := b.Delete(i)
		assert(ok)
	}
	wmCheckInvariants(b)
	assert(b.Len() == 0)
	assert(b.TotalWeight() == 0)
	assert(wmHeight(b) == 0)
}

func TestWeightedMapShuffledDeletes(t *testing.T) {
	// Random insertion and deletion order stresses every split, borrow and
	// merge path while the invariants are checked after each operation.
	rng := rand.New(rand.NewPCG(101, 202))
	b := NewWeightedMap[int, int]()
	model := make(map[int]uint64)
	const rounds = 4000
	for i := 0; i < rounds; i++ {
		key := rng.IntN(1000)
		if rng.IntN(2) == 0 {
			w := uint64(rng.IntN(8))
			_, err := b.Set(key, key, w)
			assert(err == nil)
			model[key] = w
		} else {
			_, ok := b.Delete(key)
			_, expected := model[key]
			assert(ok == expected)
			delete(model, key)
		}
		if i%250 == 0 {
			wmCheckInvariants(b)
			assert(b.Len() == len(model))
			sorted := slices.Sorted(maps.Keys(model))
			var total uint64
			for _, k := range sorted {
				total += model[k]
			}
			assert(b.TotalWeight() == total)
			for _, k := range sorted {
				assert(b.PrefixWeight(k) ==
					naivePrefixWeighted(model, sorted, k))
			}
		}
	}
	wmCheckInvariants(b)
}

func TestWeightedMapClone(t *testing.T) {
	b := NewWeightedMap[int, int]()
	for i := 0; i < 500; i++ {
		_, err := b.Set(i, i, uint64(i%4)+1)
		assert(err == nil)
	}
	original := wmCollect(b)
	c := b.Clone()
	wmCheckInvariants(c)
	// Clone shares nodes until the first write.
	assert(b.root == c.root)
	assert(c.Len() == b.Len())
	assert(c.TotalWeight() == b.TotalWeight())

	// Writes to the clone must not touch the original.
	replaced, err := c.Set(0, 999, 4)
	assert(err == nil && replaced)
	_, deleted := c.Delete(1)
	assert(deleted)
	_, err = c.Set(1000, 1000, 2)
	assert(err == nil)
	wmCheckInvariants(b)
	wmCheckInvariants(c)
	assert(b.root != c.root)
	bEntries := wmCollect(b)
	assert(len(bEntries) == len(original))
	for i := range original {
		assert(bEntries[i] == original[i])
	}
	assert(b.Get(0) == 0)
	assert(b.Contains(1))
	assert(!b.Contains(1000))
	assert(b.Len() == 500)

	cEntries := wmCollect(c)
	assert(c.Len() == 500)
	assert(c.Get(0) == 999)
	assert(!c.Contains(1))
	assert(c.Get(1000) == 1000)
	// The clone weight cache reflects its own mutations.
	var cTotal uint64
	for _, e := range cEntries {
		cTotal += e.w
	}
	assert(c.TotalWeight() == cTotal)

	// A write to the original stays local as well.
	replaced, err = b.Set(2, 888, 1)
	assert(err == nil && replaced)
	assert(c.Get(2) == 2)
	wmCheckInvariants(b)
	wmCheckInvariants(c)

	// Clone chains keep working independently.
	d := c.Clone()
	_, deleted = d.Delete(0)
	assert(deleted)
	wmCheckInvariants(d)
	wmCheckInvariants(c)
	assert(c.Contains(0))
	assert(!d.Contains(0))
	assert(b.Contains(0))
}

func TestWeightedMapRandom(t *testing.T) {
	// Several independent seeds so split, borrow, merge and overflow paths
	// are all exercised against the naive ordered model.
	weightedMapRandomOnce(t, 42, 99)
	weightedMapRandomOnce(t, 1234567, 7654321)
	weightedMapRandomOnce(t, 0xdeadbeef, 0xfeedface)
}

func weightedMapRandomOnce(t *testing.T, seed1, seed2 uint64) {
	rng := rand.New(rand.NewPCG(seed1, seed2))
	b := NewWeightedMap[int, int]()
	model := make(map[int]wmItem[int])
	const keyRange = 300
	const iters = 12000
	for iter := 0; iter < iters; iter++ {
		key := rng.IntN(keyRange)
		if rng.IntN(3) == 2 {
			// Delete.
			_, ok := b.Delete(key)
			_, expected := model[key]
			assert(ok == expected)
			delete(model, key)
		} else {
			// Set, with zero weights being very common.
			var w uint64
			switch rng.IntN(5) {
			case 0, 1:
				w = 0
			default:
				w = uint64(rng.IntN(11))
			}
			value := key*7 + 1
			old, had := model[key]
			var oldW uint64
			if had {
				oldW = old.w
			}
			overflow := false
			total := naiveTotal(model)
			if w > oldW && w-oldW > math.MaxUint64-total {
				overflow = true
			}
			replaced, err := b.Set(key, value, w)
			if overflow {
				assert(err == ErrWeightOverflow)
				assert(!replaced)
				if had {
					v, ok := b.Lookup(key)
					assert(ok && v == old.v)
				} else {
					_, ok := b.Lookup(key)
					assert(!ok)
				}
			} else {
				assert(err == nil)
				assert(replaced == had)
				model[key] = wmItem[int]{value, w}
			}
		}

		wmCheckInvariants(b)
		sorted := slices.Sorted(maps.Keys(model))
		assert(b.Len() == len(model))
		total := naiveTotal(model)
		assert(b.TotalWeight() == total)

		// Stored keys, values, weights and ordering match the model.
		entries := wmCollect(b)
		assert(len(entries) == len(model))
		for i, k := range sorted {
			assert(entries[i].k == k)
			assert(entries[i].v == model[k].v)
			assert(entries[i].w == model[k].w)
			v, ok := b.Lookup(k)
			assert(ok && v == model[k].v)
			assert(b.Contains(k))
			assert(b.PrefixWeight(k) == naivePrefix(model, sorted, k))
		}

		// Boundary offsets of every item.
		var acc uint64
		for _, k := range sorted {
			it := model[k]
			if it.w > 0 {
				wmExpectSelect(b, k, it.v, 0, acc)
				wmExpectSelect(b, k, it.v, it.w-1, acc+it.w-1)
			}
			acc += it.w
		}
		assert(acc == total)

		// Periodically scan every legal offset against the naive model.
		if iter%25 == 0 && total > 0 {
			for off := uint64(0); off < total; off++ {
				k, v, itemOffset, ok := b.SelectWeight(off)
				assert(ok)
				ek, eio, eok := naiveSelect(model, sorted, off)
				assert(eok)
				assert(k == ek)
				assert(v == model[ek].v)
				assert(itemOffset == eio)
			}
		}

		// The total offset never hits.
		_, _, _, ok := b.SelectWeight(total)
		assert(!ok)

		// Random valid offsets plus absent-key prefix probes.
		if total > 0 {
			for range 8 {
				off := rng.Uint64N(total)
				k, _, itemOffset, ok := b.SelectWeight(off)
				assert(ok)
				ek, eio, eok := naiveSelect(model, sorted, off)
				assert(eok && k == ek && itemOffset == eio)
			}
		}
		for _, probe := range []int{-100, -1, keyRange, keyRange + 100,
			rng.IntN(keyRange)} {
			assert(b.PrefixWeight(probe) ==
				naivePrefix(model, sorted, probe))
		}
	}
}

func TestWeightedMapVisitCounts(t *testing.T) {
	build := func(n int) *WeightedMap[int, int] {
		b := NewWeightedMap[int, int]()
		for i := 0; i < n; i++ {
			w := uint64(1)
			if i%5 == 0 {
				w = 0
			}
			_, err := b.Set(i, i, w)
			assert(err == nil)
		}
		wmCheckInvariants(b)
		return b
	}
	small := build(2000)
	big := build(20000)
	hs := wmHeight(small)
	hb := wmHeight(big)
	assert(hs >= 1)
	assert(hb > hs)

	maxVisits := func(b *WeightedMap[int, int]) (int, int) {
		maxPrefix, maxSelect := 0, 0
		for k := 0; k < b.Len(); k += 7 {
			_, visits := b.prefixWeightCounted(k)
			if visits > maxPrefix {
				maxPrefix = visits
			}
		}
		total := b.TotalWeight()
		for off := uint64(0); off < total; off += 97 {
			_, _, _, ok, visits := b.selectWeightCounted(off)
			assert(ok)
			if visits > maxSelect {
				maxSelect = visits
			}
		}
		_, _, _, ok, visits := b.selectWeightCounted(total - 1)
		assert(ok)
		if visits > maxSelect {
			maxSelect = visits
		}
		return maxPrefix, maxSelect
	}

	sp, ss := maxVisits(small)
	bp, bs := maxVisits(big)
	// Every query visits at most one node per tree level.
	assert(sp <= hs+1)
	assert(ss <= hs+1)
	assert(bp <= hb+1)
	assert(bs <= hb+1)
	// A tenfold increase in items must not make the queries scan linearly.
	assert(bp <= sp+1)
	assert(bs <= ss+1)
	assert(bp < 10)
	assert(bs < 10)
}

func TestWeightedMapStringKeys(t *testing.T) {
	b := NewWeightedMap[string, int]()
	index := map[string]int{
		"banana": 0, "apple": 1, "cherry": 2, "": 3,
		"avocado": 4, "date": 5,
	}
	for k, idx := range index {
		_, err := b.Set(k, idx, uint64(idx+1))
		assert(err == nil)
	}
	wmCheckInvariants(b)
	sorted := slices.Sorted(maps.Keys(index))
	var acc uint64
	for _, k := range sorted {
		assert(b.PrefixWeight(k) == acc)
		v, ok := b.Lookup(k)
		assert(ok && v == index[k])
		acc += uint64(index[k] + 1)
	}
	assert(b.TotalWeight() == acc)
	// Smoke check: offset zero always lands on the first item with a zero
	// in-item offset.
	k, _, itemOffset, ok := b.SelectWeight(0)
	assert(ok && k == sorted[0] && itemOffset == 0)
}
