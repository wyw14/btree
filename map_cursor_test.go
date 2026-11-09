// https://github.com/tidwall/btree
//
// Copyright 2026 Joshua J Baker. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package btree

import (
	"math/rand/v2"
	"slices"
	"testing"
)

func mustOK(t *testing.T, ok bool, err error, what string) {
	t.Helper()
	if err != nil || !ok {
		t.Fatalf("%s: ok=%v err=%v", what, ok, err)
	}
}

func mustMiss(t *testing.T, ok bool, err error, what string) {
	t.Helper()
	if err != nil || ok {
		t.Fatalf("%s: expected miss, ok=%v err=%v", what, ok, err)
	}
}

// curExpectInt asserts the full cursor state against model position p
// (-1 before-first, len(keys) after-last) for an int-keyed map whose values
// are key*10.
func curExpectInt(t *testing.T, c *MapCursor[int, int], keys []int, p int) {
	t.Helper()
	if err := c.Err(); err != nil {
		t.Fatalf("cursor unexpectedly invalid: %v", err)
	}
	n := len(keys)
	switch {
	case p == -1:
		if c.Valid() || !c.BeforeFirst() || c.AfterLast() {
			t.Fatalf("want before-first, got valid=%v before=%v after=%v",
				c.Valid(), c.BeforeFirst(), c.AfterLast())
		}
	case p == n:
		if c.Valid() || c.BeforeFirst() || !c.AfterLast() {
			t.Fatalf("want after-last, got valid=%v before=%v after=%v",
				c.Valid(), c.BeforeFirst(), c.AfterLast())
		}
	default:
		if !c.Valid() || c.BeforeFirst() || c.AfterLast() {
			t.Fatalf("want valid at %d, got valid=%v before=%v after=%v",
				p, c.Valid(), c.BeforeFirst(), c.AfterLast())
		}
		key, value, ok := c.Item()
		if !ok || key != keys[p] || value != key*10 {
			t.Fatalf("at %d want item %v:%d, got %v:%d ok=%v",
				p, keys[p], keys[p]*10, key, value, ok)
		}
		if k, ok := c.Key(); !ok || k != keys[p] {
			t.Fatalf("Key at %d: got %v ok=%v", p, k, ok)
		}
		if v, ok := c.Value(); !ok || v != keys[p]*10 {
			t.Fatalf("Value at %d: got %v ok=%v", p, v, ok)
		}
		if idx, ok := c.Index(); !ok || idx != p {
			t.Fatalf("Index at %d: got %d ok=%v", p, idx, ok)
		}
	}
	if p == -1 || p == n {
		// Sentinel reads must never masquerade as a stored item, even when
		// the zero key/value is a legitimate item.
		if k, _, ok := c.Item(); ok || k != 0 {
			t.Fatalf("sentinel Item must report false with zero key, got %v ok=%v",
				k, ok)
		}
		if _, ok := c.Key(); ok {
			t.Fatalf("sentinel Key must report false")
		}
		if _, ok := c.Value(); ok {
			t.Fatalf("sentinel Value must report false")
		}
		if _, ok := c.Index(); ok {
			t.Fatalf("sentinel Index must report false")
		}
	}
}

func newIntMap(n int) (*Map[int, int], []int) {
	b := NewMap[int, int]()
	order := rand.New(rand.NewPCG(1, 2)).Perm(n)
	for _, i := range order {
		b.Set(i, i*10)
	}
	keys := make([]int, n)
	for i := range keys {
		keys[i] = i
	}
	return b, keys
}

func TestMapCursorEmptyOps(t *testing.T) {
	b := NewMap[int, int]()

	// A fresh cursor starts before-first.
	c := b.Cursor()
	if c.Valid() || !c.BeforeFirst() || c.AfterLast() {
		t.Fatalf("new cursor should be before-first")
	}
	if err := c.Err(); err != nil {
		t.Fatalf("fresh cursor error: %v", err)
	}

	// First and Last both miss on an empty tree.
	ok, err := c.First()
	mustMiss(t, ok, err, "First on empty")
	if !c.AfterLast() {
		t.Fatalf("First on empty should land after-last")
	}
	c = b.Cursor()
	ok, err = c.Last()
	mustMiss(t, ok, err, "Last on empty")
	if !c.BeforeFirst() {
		t.Fatalf("Last on empty should land before-first")
	}

	// Next from before-first reaches after-last on an empty tree, and Prev
	// from after-last reaches before-first; the sentinels then stay put.
	c = b.Cursor()
	ok, err = c.Next()
	mustMiss(t, ok, err, "Next on empty")
	if !c.AfterLast() {
		t.Fatalf("Next on empty should land after-last")
	}
	ok, err = c.Prev()
	mustMiss(t, ok, err, "Prev on empty")
	if !c.BeforeFirst() {
		t.Fatalf("Prev on empty should land before-first")
	}
	ok, err = c.Prev()
	mustMiss(t, ok, err, "Prev at before")
	if !c.BeforeFirst() {
		t.Fatalf("Prev at before-first should stay")
	}
	ok, err = c.Next()
	mustMiss(t, ok, err, "Next toward after")
	ok, err = c.Next()
	mustMiss(t, ok, err, "Next at after")
	if !c.AfterLast() {
		t.Fatalf("Next at after-last should stay")
	}

	// SeekGE and SeekIndex bounds on an empty tree.
	if ok, err := c.SeekGE(0); ok || err != nil || !c.AfterLast() {
		t.Fatalf("SeekGE on empty: ok=%v err=%v", ok, err)
	}
	if ok, err := c.SeekIndex(0); ok || err != nil || !c.AfterLast() {
		t.Fatalf("SeekIndex(0) on empty should be after-last")
	}
	if ok, err := c.SeekIndex(-5); ok || err != nil || !c.BeforeFirst() {
		t.Fatalf("SeekIndex(-5) on empty should be before-first")
	}
	if ok, err := c.SeekIndex(5); ok || err != nil || !c.AfterLast() {
		t.Fatalf("SeekIndex(5) on empty should be after-last")
	}
	if _, _, ok := c.Item(); ok {
		t.Fatalf("Item on empty must miss")
	}
}

func TestMapCursorSingleAndZeroItem(t *testing.T) {
	b := NewMap[int, int]()
	b.Set(0, 0) // legitimate zero key/value
	c := b.Cursor()
	if _, _, ok := c.Item(); ok {
		t.Fatalf("before-first must not expose the zero item")
	}
	ok, err := c.Next()
	mustOK(t, ok, err, "next into zero item")
	k, v, ok := c.Item()
	if !ok || k != 0 || v != 0 {
		t.Fatalf("zero item: got %v:%v ok=%v", k, v, ok)
	}
	idx, ok := c.Index()
	if !ok || idx != 0 {
		t.Fatalf("zero item index: %d ok=%v", idx, ok)
	}
	ok, err = c.Next()
	mustMiss(t, ok, err, "Next past only item")
	if !c.AfterLast() {
		t.Fatal("expected after-last")
	}
	ok, err = c.First()
	mustOK(t, ok, err, "First on single")
	ok, err = c.Last()
	mustOK(t, ok, err, "Last on single")
	if k, _, _ := c.Item(); k != 0 {
		t.Fatalf("single item key mismatch")
	}
	ok, err = c.Prev()
	mustMiss(t, ok, err, "Prev before only item")
	if !c.BeforeFirst() {
		t.Fatal("expected before-first")
	}

	b = NewMap[int, int]()
	b.Set(42, 420)
	c = b.Cursor()
	ok, err = c.First()
	mustOK(t, ok, err, "First")
	curExpectInt(t, c, []int{42}, 0)
	ok, err = c.Next()
	mustMiss(t, ok, err, "Next past only item")
	ok, err = c.Prev()
	mustOK(t, ok, err, "Prev from after")
	curExpectInt(t, c, []int{42}, 0)
}

func TestMapCursorFullScanAndSteps(t *testing.T) {
	const n = 5000
	b, keys := newIntMap(n)
	if b.tree.Height() < 2 {
		t.Fatalf("test tree must span multiple levels, height=%d",
			b.tree.Height())
	}

	c := b.Cursor()
	ok, err := c.First()
	mustOK(t, ok, err, "First")
	curExpectInt(t, c, keys, 0)

	startSteps := c.steps
	got := make([]int, 0, n)
	for {
		k, _, okk := c.Item()
		if !okk {
			t.Fatalf("item should be valid before after-last")
		}
		got = append(got, k)
		idx, okk := c.Index()
		if !okk || idx != len(got)-1 {
			t.Fatalf("index mismatch: %d vs %d", idx, len(got)-1)
		}
		cont, err := c.Next()
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !cont {
			break
		}
	}
	if !slices.Equal(got, keys) {
		t.Fatalf("forward scan mismatch")
	}
	if !c.AfterLast() {
		t.Fatalf("should be after-last after full scan")
	}

	// Amortized-constant stepping: walking all n items via Next must cost
	// O(n) primitive steps. A re-search from the root per item would cost
	// at least n*height steps; enforce a strict linear bound instead.
	fwdSteps := c.steps - startSteps
	if fwdSteps > 2*n {
		t.Fatalf("forward scan took %d steps (>2*%d); not amortized constant",
			fwdSteps, n)
	}

	// Backward scan from Last with the same bound.
	ok, err = c.Last()
	mustOK(t, ok, err, "Last")
	startSteps = c.steps
	for i := n - 1; i >= 0; i-- {
		k, _, okk := c.Item()
		idx, iok := c.Index()
		if !okk || !iok || k != keys[i] || idx != i {
			t.Fatalf("backward at %d: key=%d idx=%d ok=%v", i, k, idx, okk)
		}
		cont, err := c.Prev()
		if err != nil {
			t.Fatalf("Prev: %v", err)
		}
		if i == 0 {
			if cont || !c.BeforeFirst() {
				t.Fatalf("last Prev should reach before-first")
			}
		} else if !cont {
			t.Fatalf("Prev missed at %d", i)
		}
	}
	backSteps := c.steps - startSteps
	if backSteps > 2*n {
		t.Fatalf("backward scan took %d steps (>2*%d)", backSteps, n)
	}

	// Prev from after-last goes straight to the final item; Next from
	// before-first goes straight to the first item.
	ok, err = c.SeekIndex(n + 1)
	mustMiss(t, ok, err, "seek past end")
	ok, err = c.Prev()
	mustOK(t, ok, err, "Prev from after-last")
	curExpectInt(t, c, keys, n-1)
	ok, err = c.SeekIndex(-1)
	mustMiss(t, ok, err, "seek before start")
	ok, err = c.Next()
	mustOK(t, ok, err, "Next from before-first")
	curExpectInt(t, c, keys, 0)
}

func TestMapCursorAfterDeletes(t *testing.T) {
	// Build a tree then delete every even key: the resulting nodes are
	// near minItems and heavily rebalanced, exercising separator walks.
	const n = 6000
	b := NewMap[int, int]()
	for i := 0; i < 2*n; i++ {
		b.Set(i, i*10)
	}
	for i := 0; i < 2*n; i += 2 {
		if _, ok := b.Delete(i); !ok {
			t.Fatalf("delete %d", i)
		}
	}
	if b.Len() != n {
		t.Fatalf("len=%d", b.Len())
	}
	keys := make([]int, n)
	for i := range keys {
		keys[i] = 2*i + 1
	}
	c := b.Cursor()
	ok, err := c.First()
	mustOK(t, ok, err, "First")
	start := c.steps
	for i := 0; i < n; i++ {
		k, _, okk := c.Item()
		idx, iok := c.Index()
		if !okk || !iok || k != keys[i] || idx != i {
			t.Fatalf("at %d got key=%d idx=%d", i, k, idx)
		}
		cont, err := c.Next()
		if err != nil {
			t.Fatal(err)
		}
		if i == n-1 {
			if cont || !c.AfterLast() {
				t.Fatal("should end after-last")
			}
		} else if !cont {
			t.Fatalf("early stop at %d", i)
		}
	}
	if steps := c.steps - start; steps > 2*n {
		t.Fatalf("scan steps=%d > 2*%d", steps, n)
	}
	// Backward.
	ok, _ = c.Last()
	mustOK(t, ok, nil, "Last")
	for i := n - 1; i >= 0; i-- {
		k, _, okk := c.Item()
		idx, iok := c.Index()
		if !okk || !iok || k != keys[i] || idx != i {
			t.Fatalf("back at %d: key=%d idx=%d", i, k, idx)
		}
		if i > 0 {
			if ok, _ := c.Prev(); !ok {
				t.Fatalf("Prev at %d", i)
			}
		}
	}
	if ok, _ := c.Prev(); ok || !c.BeforeFirst() {
		t.Fatal("should reach before-first")
	}
	// SeekGE must skip deleted gaps.
	for q := 0; q < 2*n+5; q++ {
		ok, err := c.SeekGE(q)
		sk, _, sok := b.Seek(q)
		if err != nil {
			t.Fatal(err)
		}
		if ok != sok {
			t.Fatalf("SeekGE(%d) ok=%v map=%v", q, ok, sok)
		}
		if ok {
			if k, _, _ := c.Item(); k != sk {
				t.Fatalf("SeekGE(%d) key=%d map=%d", q, k, sk)
			}
		}
	}
}

func TestMapCursorRandomWalk(t *testing.T) {
	for _, seed := range [][2]uint64{{9, 17}, {101, 202}, {777, 888}} {
		t.Run("", func(t *testing.T) {
			cursorRandomWalk(t, 2000, seed, 20000, 1)
		})
	}
	// Deep tree (height >= 3) so steps cross multiple internal levels.
	t.Run("deep", func(t *testing.T) {
		cursorRandomWalk(t, 270000, [2]uint64{42, 43}, 2000, 3)
	})
}

func cursorRandomWalk(t *testing.T, n int, seed [2]uint64, iters, minHeight int) {
	b, keys := newIntMap(n)
	if b.tree.Height() < minHeight {
		t.Fatalf("height=%d want >=%d", b.tree.Height(), minHeight)
	}
	rng := rand.New(rand.NewPCG(seed[0], seed[1]))
	c := b.Cursor()
	p := -1
	curExpectInt(t, c, keys, p)
	for iter := 0; iter < iters; iter++ {
		switch rng.IntN(10) {
		case 0:
			ok, _ := c.First()
			if !ok {
				t.Fatal("First")
			}
			p = 0
		case 1:
			ok, _ := c.Last()
			if !ok {
				t.Fatal("Last")
			}
			p = n - 1
		case 2:
			key := rng.IntN(n+10) - 5
			ok, _ := c.SeekGE(key)
			switch {
			case key < 0:
				p = 0
			case key >= n:
				p = n
			default:
				p = key
			}
			if (p >= 0 && p < n) != ok {
				t.Fatalf("SeekGE(%d) ok=%v p=%d", key, ok, p)
			}
		case 3:
			idx := rng.IntN(n+10) - 5
			ok, _ := c.SeekIndex(idx)
			switch {
			case idx < 0:
				p = -1
			case idx >= n:
				p = n
			default:
				p = idx
			}
			if (p >= 0 && p < n) != ok {
				t.Fatalf("SeekIndex(%d) ok=%v p=%d", idx, ok, p)
			}
		case 4:
			ok, _ := c.Next()
			if p < n {
				p++
			}
			if (p >= 0 && p < n) != ok {
				t.Fatalf("Next ok=%v p=%d", ok, p)
			}
		case 5:
			ok, _ := c.Prev()
			if p > -1 {
				p--
			}
			if (p >= 0 && p < n) != ok {
				t.Fatalf("Prev ok=%v p=%d", ok, p)
			}
		case 6, 7:
			var delta int
			switch rng.IntN(6) {
			case 0:
				delta = cursorMaxInt
			case 1:
				delta = cursorMinInt
			case 2:
				delta = rng.IntN(10) - 5
			case 3:
				delta = 0
			case 4:
				delta = rng.IntN(n + 50)
			default:
				delta = -rng.IntN(n + 50)
			}
			ok, _ := c.Skip(delta)
			switch {
			case delta > 0:
				if delta >= n-p {
					p = n
				} else {
					p += delta
				}
			case delta < 0:
				if delta == cursorMinInt || -delta >= p+1 {
					p = -1
				} else {
					p += delta
				}
			}
			if (p >= 0 && p < n) != ok {
				t.Fatalf("Skip(%d) ok=%v p=%d", delta, ok, p)
			}
		case 8:
			// One Next then one Prev, modeled exactly (including the
			// asymmetric sentinel rules: Next at after-last stays there,
			// Prev at before-first stays there).
			before := p
			ok1, _ := c.Next()
			if p < n {
				p++
			}
			ok2, _ := c.Prev()
			if p > -1 {
				p--
			}
			// ok iff the post-move position is a valid item.
			wantOK1 := before < n-1 // -1->0 valid, n-1->n sentinel
			if ok1 != wantOK1 {
				t.Fatalf("Next flag at %d: got %v want %v",
					before, ok1, wantOK1)
			}
			afterNext := before
			if afterNext < n {
				afterNext++
			}
			wantOK2 := afterNext >= 1 // 0->-1 sentinel, n->n-1 valid
			if ok2 != wantOK2 {
				t.Fatalf("Prev flag at %d: got %v want %v",
					afterNext, ok2, wantOK2)
			}
		case 9:
			// Run Next to the end, then Prev the same number of steps.
			forward := 0
			for {
				cont, err := c.Next()
				if err != nil {
					t.Fatal(err)
				}
				if !cont {
					break
				}
				forward++
			}
			if !c.AfterLast() {
				t.Fatal("run should end after-last")
			}
			if p < n {
				for i := 0; i < forward; i++ {
					cont, err := c.Prev()
					if err != nil || !cont {
						t.Fatalf("rewind at %d: ok=%v err=%v", i, cont, err)
					}
				}
				// One more step restores the starting position: it lands
				// on an item when p>=0, and on before-first when p==-1.
				cont, err := c.Prev()
				if err != nil || cont != (p >= 0) {
					t.Fatalf("final rewind at p=%d: ok=%v err=%v",
						p, cont, err)
				}
			}
		}
		curExpectInt(t, c, keys, p)
	}
}

func TestMapCursorSeekGEGaps(t *testing.T) {
	b := NewMap[int, int]()
	for i := 0; i < 100; i++ {
		b.Set(i*7, i)
	}
	c := b.Cursor()
	for q := -10; q <= 7*100; q++ {
		ok, err := c.SeekGE(q)
		sk, sv, sok := b.Seek(q)
		if err != nil {
			t.Fatal(err)
		}
		if ok != sok {
			t.Fatalf("SeekGE(%d): ok=%v map.Seek=%v", q, ok, sok)
		}
		if ok {
			ck, cv, cok := c.Item()
			if !cok || ck != sk || cv != sv {
				t.Fatalf("SeekGE(%d): cursor %v:%v map %v:%v",
					q, ck, cv, sk, sv)
			}
		} else if !c.AfterLast() {
			t.Fatalf("SeekGE(%d) miss should be after-last", q)
		}
	}
}

func TestMapCursorSkipExtremes(t *testing.T) {
	const n = 100
	b, keys := newIntMap(n)
	c := b.Cursor()

	ok, err := c.Skip(cursorMaxInt)
	mustMiss(t, ok, err, "huge positive skip")
	if !c.AfterLast() {
		t.Fatal("huge positive skip should stop after-last")
	}
	ok, err = c.Skip(cursorMinInt)
	mustMiss(t, ok, err, "MinInt skip")
	if !c.BeforeFirst() {
		t.Fatal("MinInt skip from after-last should stop before-first")
	}
	ok, err = c.Skip(n * 10)
	mustMiss(t, ok, err, "overshoot forward")
	if !c.AfterLast() {
		t.Fatal("overshoot forward lands after")
	}
	// One step back from after-last enters the final item.
	ok, err = c.Skip(-1)
	mustOK(t, ok, err, "Skip(-1) from after")
	curExpectInt(t, c, keys, n-1)
	// Exactly n steps from before-first lands on the after-last sentinel.
	ok, err = c.Skip(cursorMinInt)
	mustMiss(t, ok, err, "reset to before")
	if !c.BeforeFirst() {
		t.Fatal("MinInt reset should land before-first")
	}
	// From before-first (-1), n steps land on the last item (n-1), and
	// n+1 steps land on after-last (n).
	ok, err = c.Skip(n)
	mustOK(t, ok, err, "Skip(n) from before")
	curExpectInt(t, c, keys, n-1)
	ok, err = c.Skip(1)
	mustMiss(t, ok, err, "Skip(1) to after")
	if !c.AfterLast() {
		t.Fatal("expected after-last")
	}
	ok, err = c.Skip(cursorMinInt)
	mustMiss(t, ok, err, "reset before")
	ok, err = c.Skip(n + 1)
	mustMiss(t, ok, err, "Skip(n+1) to after")
	if !c.AfterLast() {
		t.Fatal("Skip(n+1) from before must land after-last")
	}
	// One back returns to the last item.
	ok, err = c.Skip(-1)
	mustOK(t, ok, err, "Skip(-1) back")
	curExpectInt(t, c, keys, n-1)
	// Exact steps back to first item and then before-first.
	ok, err = c.Skip(-(n - 1))
	mustOK(t, ok, err, "Skip to first")
	curExpectInt(t, c, keys, 0)
	ok, err = c.Skip(-1)
	mustMiss(t, ok, err, "Skip to before")
	if !c.BeforeFirst() {
		t.Fatal("expected before-first")
	}
	// Skip(0) reports state without moving.
	ok, err = c.Skip(0)
	mustMiss(t, ok, err, "Skip(0) at before")
	if !c.BeforeFirst() {
		t.Fatal("Skip(0) must not move")
	}
	ok, err = c.Next()
	mustOK(t, ok, err, "Next")
	ok, err = c.Skip(0)
	mustOK(t, ok, err, "Skip(0) on item")
}

func TestMapCursorStringPrefix(t *testing.T) {
	t.Run("prefix", func(t *testing.T) {
		testStringCursor(t, NewMap[string, int]())
	})
	t.Run("noprefix", func(t *testing.T) {
		testStringCursor(t, NewMapOptions[string, int](MapOptions[string, int]{
			NoPrefix: true,
		}))
	})
}

func testStringCursor(t *testing.T, b *Map[string, int]) {
	const n = 1500
	keys := make([]string, n)
	for i := 0; i < n; i++ {
		keys[i] = stringKey(i)
	}
	rng := rand.New(rand.NewPCG(3, 4))
	for _, i := range rng.Perm(n) {
		b.Set(keys[i], i*10)
	}
	if !slices.IsSorted(keys) {
		t.Fatal("model keys must be sorted")
	}
	c := b.Cursor()
	p := -1
	curExpectString(t, c, keys, p)
	rng2 := rand.New(rand.NewPCG(5, 6))
	for iter := 0; iter < 10000; iter++ {
		switch rng2.IntN(7) {
		case 0:
			ok, _ := c.First()
			if !ok {
				t.Fatal("First")
			}
			p = 0
		case 1:
			ok, _ := c.Last()
			if !ok {
				t.Fatal("Last")
			}
			p = n - 1
		case 2:
			i := rng2.IntN(n+4) - 2
			var key string
			switch {
			case i < 0:
				key = "aa"
			case i >= n:
				key = "zz"
			default:
				key = keys[i]
			}
			ok, _ := c.SeekGE(key)
			switch key {
			case "aa":
				p = 0
			case "zz":
				p = n
			default:
				p = i
			}
			if (p >= 0 && p < n) != ok {
				t.Fatalf("SeekGE ok=%v p=%d", ok, p)
			}
		case 3:
			ok, _ := c.Next()
			if p < n {
				p++
			}
			if (p >= 0 && p < n) != ok {
				t.Fatalf("Next ok=%v p=%d", ok, p)
			}
		case 4:
			ok, _ := c.Prev()
			if p > -1 {
				p--
			}
			if (p >= 0 && p < n) != ok {
				t.Fatalf("Prev ok=%v p=%d", ok, p)
			}
		case 5:
			idx := rng2.IntN(n+4) - 2
			ok, _ := c.SeekIndex(idx)
			switch {
			case idx < 0:
				p = -1
			case idx >= n:
				p = n
			default:
				p = idx
			}
			if (p >= 0 && p < n) != ok {
				t.Fatalf("SeekIndex ok=%v p=%d", ok, p)
			}
		case 6:
			delta := rng2.IntN(20) - 10
			ok, _ := c.Skip(delta)
			p += delta
			if p < -1 {
				p = -1
			}
			if p > n {
				p = n
			}
			if (p >= 0 && p < n) != ok {
				t.Fatalf("Skip(%d) ok=%v p=%d", delta, ok, p)
			}
		}
		curExpectString(t, c, keys, p)
	}
}

func stringKey(i int) string {
	const digits = "0123456789"
	var buf [8]byte
	for j := 7; j >= 0; j-- {
		buf[j] = digits[i%10]
		i /= 10
	}
	return "key:" + string(buf[:])
}

func curExpectString(t *testing.T, c *MapCursor[string, int], keys []string, p int) {
	t.Helper()
	n := len(keys)
	valid := p >= 0 && p < n
	if c.Valid() != valid {
		t.Fatalf("Valid=%v want %v (p=%d)", c.Valid(), valid, p)
	}
	if p == -1 && !c.BeforeFirst() {
		t.Fatalf("want before-first")
	}
	if p == n && !c.AfterLast() {
		t.Fatalf("want after-last")
	}
	if valid {
		k, v, ok := c.Item()
		if !ok || k != keys[p] || v != p*10 {
			t.Fatalf("at %d want %q:%d got %q:%d ok=%v",
				p, keys[p], p*10, k, v, ok)
		}
		idx, ok := c.Index()
		if !ok || idx != p {
			t.Fatalf("Index=%d ok=%v want %d", idx, ok, p)
		}
	}
}

func TestMapCursorInvalidationTable(t *testing.T) {
	const n = 200

	type tc struct {
		name   string
		change bool
		fn     func(b *Map[int, int])
	}
	cases := []tc{
		{"Insert", true, func(b *Map[int, int]) { b.Insert(-1, -10) }},
		{"InsertDuplicate", false, func(b *Map[int, int]) { b.Insert(5, 0) }},
		{"Replace", true, func(b *Map[int, int]) { b.Replace(5, 55) }},
		{"ReplaceMiss", false, func(b *Map[int, int]) { b.Replace(999, 1) }},
		{"SetReplace", true, func(b *Map[int, int]) { b.Set(6, 66) }},
		{"SetInsert", true, func(b *Map[int, int]) { b.Set(-2, -20) }},
		{"Delete", true, func(b *Map[int, int]) { b.Delete(7) }},
		{"DeleteMiss", false, func(b *Map[int, int]) { b.Delete(999) }},
		{"ActionReplace", true, func(b *Map[int, int]) {
			b.Action(8, func(v int) (int, Action) { return v + 1, ActionReplace })
		}},
		{"ActionDelete", true, func(b *Map[int, int]) {
			b.Action(9, func(v int) (int, Action) { return v, ActionDelete })
		}},
		{"ActionGet", false, func(b *Map[int, int]) {
			b.Action(10, func(v int) (int, Action) { return v, ActionGet })
		}},
		{"ActionMiss", false, func(b *Map[int, int]) {
			b.Action(999, func(v int) (int, Action) { return v, ActionReplace })
		}},
		{"Clear", true, func(b *Map[int, int]) { b.Clear() }},
		{"DrainAt", true, func(b *Map[int, int]) { b.DrainAt(10, 5) }},
		{"DrainAtEmpty", false, func(b *Map[int, int]) { b.DrainAt(100, 0) }},
		{"Drain", true, func(b *Map[int, int]) { b.Drain(20, 25) }},
		{"DrainMiss", false, func(b *Map[int, int]) { b.Drain(300, 400) }},
		{"DrainAtOptions", true, func(b *Map[int, int]) {
			b.DrainAtOptions(0, 3, DrainOptions{NoReturn: true})
		}},
		{"DrainOptions", true, func(b *Map[int, int]) {
			b.DrainOptions(30, 33, DrainOptions{MaxInclusive: true})
		}},
		{"PushFront", true, func(b *Map[int, int]) { b.PushFront(-3, -30) }},
		{"PushFrontOutOfOrder", false, func(b *Map[int, int]) {
			b.PushFront(50, 0)
		}},
		{"PushBack", true, func(b *Map[int, int]) { b.PushBack(200, 2000) }},
		{"PushBackOutOfOrder", false, func(b *Map[int, int]) {
			b.PushBack(50, 0)
		}},
		{"PopFront", true, func(b *Map[int, int]) { b.PopFront() }},
		{"PopBack", true, func(b *Map[int, int]) { b.PopBack() }},
		{"PopFrontFuncYes", true, func(b *Map[int, int]) {
			b.PopFrontFunc(func(k, v int) bool { return true })
		}},
		{"PopFrontFuncNo", false, func(b *Map[int, int]) {
			b.PopFrontFunc(func(k, v int) bool { return false })
		}},
		{"PopBackFuncYes", true, func(b *Map[int, int]) {
			b.PopBackFunc(func(k, v int) bool { return true })
		}},
		{"PopBackFuncNo", false, func(b *Map[int, int]) {
			b.PopBackFunc(func(k, v int) bool { return false })
		}},
		{"InsertAt", true, func(b *Map[int, int]) { b.InsertAt(0, -4, -40) }},
		{"InsertAtOOB", false, func(b *Map[int, int]) {
			b.InsertAt(n+1, -4, -40)
		}},
		{"DeleteAt", true, func(b *Map[int, int]) { b.DeleteAt(50) }},
		{"DeleteAtOOB", false, func(b *Map[int, int]) { b.DeleteAt(n) }},
		{"DeleteAtNeg", false, func(b *Map[int, int]) { b.DeleteAt(-1) }},
		{"ReplaceAt", true, func(b *Map[int, int]) { b.ReplaceAt(50, 50, 500) }},
		{"ReplaceAtOOB", false, func(b *Map[int, int]) {
			b.ReplaceAt(n, 0, 0)
		}},
		{"DeleteFunc", true, func(b *Map[int, int]) {
			b.DeleteFunc(func(k, v int) bool { return k == 77 })
		}},
		{"DeleteFuncNone", false, func(b *Map[int, int]) {
			b.DeleteFunc(func(k, v int) bool { return false })
		}},
		{"LookupMut", true, func(b *Map[int, int]) { b.LookupMut(80) }},
		{"LookupMutMiss", false, func(b *Map[int, int]) { b.LookupMut(999) }},
		{"SeekMut", true, func(b *Map[int, int]) { b.SeekMut(81) }},
		{"SeekMutMiss", false, func(b *Map[int, int]) { b.SeekMut(999) }},
		{"SeekNextMut", true, func(b *Map[int, int]) { b.SeekNextMut(81) }},
		{"SeekNextMutMiss", false, func(b *Map[int, int]) {
			b.SeekNextMut(n - 1)
		}},
		{"SeekPrevMut", true, func(b *Map[int, int]) { b.SeekPrevMut(81) }},
		{"SeekPrevMutMiss", false, func(b *Map[int, int]) { b.SeekPrevMut(0) }},
		{"FrontMut", true, func(b *Map[int, int]) { b.FrontMut() }},
		{"BackMut", true, func(b *Map[int, int]) { b.BackMut() }},
		{"LookupAtMut", true, func(b *Map[int, int]) { b.LookupAtMut(90) }},
		{"LookupAtMutOOB", false, func(b *Map[int, int]) { b.LookupAtMut(n) }},
		{"GetAtMut", true, func(b *Map[int, int]) { b.GetAtMut(91) }},
		{"GetAtMutOOB", false, func(b *Map[int, int]) { b.GetAtMut(n) }},
		{"AllMutIter", true, func(b *Map[int, int]) {
			for range b.AllMut() {
				break
			}
		}},
		{"BackwardMutIter", true, func(b *Map[int, int]) {
			for range b.BackwardMut() {
				break
			}
		}},
		{"AscendMutIter", true, func(b *Map[int, int]) {
			for range b.AscendMut(0) {
				break
			}
		}},
		{"DescendMutIter", true, func(b *Map[int, int]) {
			for range b.DescendMut(1000) {
				break
			}
		}},
		{"AscendAtMutIter", true, func(b *Map[int, int]) {
			for range b.AscendAtMut(5) {
				break
			}
		}},
		{"AscendAtMutOOB", false, func(b *Map[int, int]) {
			for range b.AscendAtMut(n) {
				t.Fatal("should yield nothing")
			}
		}},
		{"DescendAtMutIter", true, func(b *Map[int, int]) {
			for range b.DescendAtMut(5) {
				break
			}
		}},
		{"ValuesMutIter", true, func(b *Map[int, int]) {
			for range b.ValuesMut() {
				break
			}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, keys := newIntMap(n)
			c := b.Cursor()
			ok, err := c.SeekGE(100)
			mustOK(t, ok, err, "setup SeekGE")
			curExpectInt(t, c, keys, 100)
			tc.fn(b)
			if tc.change {
				if err := c.Err(); err != ErrMapCursorInvalid {
					t.Fatalf("%s: want invalidation, err=%v", tc.name, err)
				}
				assertCursorDead(t, c, tc.name)
			} else {
				if err := c.Err(); err != nil {
					t.Fatalf("%s: unexpected invalidation: %v", tc.name, err)
				}
				curExpectInt(t, c, keys, 100)
			}
		})
	}
}

func assertCursorDead(t *testing.T, c *MapCursor[int, int], name string) {
	t.Helper()
	if c.Valid() || c.BeforeFirst() || c.AfterLast() {
		t.Fatalf("%s: dead cursor must report no state", name)
	}
	if _, _, ok := c.Item(); ok {
		t.Fatalf("%s: dead cursor Item", name)
	}
	if _, ok := c.Key(); ok {
		t.Fatalf("%s: dead cursor Key", name)
	}
	if _, ok := c.Value(); ok {
		t.Fatalf("%s: dead cursor Value", name)
	}
	if _, ok := c.Index(); ok {
		t.Fatalf("%s: dead cursor Index", name)
	}
	type move func() (bool, error)
	for _, mv := range []move{
		c.First, c.Last, c.Next, c.Prev,
		func() (bool, error) { return c.SeekGE(1) },
		func() (bool, error) { return c.SeekIndex(1) },
		func() (bool, error) { return c.Skip(1) },
	} {
		ok, err := mv()
		if ok || err != ErrMapCursorInvalid {
			t.Fatalf("%s: dead move ok=%v err=%v", name, ok, err)
		}
	}
	if err := c.Err(); err != ErrMapCursorInvalid {
		t.Fatalf("%s: Err not stable: %v", name, err)
	}
}

func TestMapCursorEmptyMutNoInvalidation(t *testing.T) {
	b := NewMap[int, int]()
	c := b.Cursor()
	if _, ok := b.LookupMut(1); ok {
		t.Fatal("miss")
	}
	if _, _, ok := b.SeekMut(1); ok {
		t.Fatal("miss")
	}
	if _, _, ok := b.SeekNextMut(1); ok {
		t.Fatal("miss")
	}
	if _, _, ok := b.SeekPrevMut(1); ok {
		t.Fatal("miss")
	}
	if _, _, ok := b.FrontMut(); ok {
		t.Fatal("miss")
	}
	if _, _, ok := b.BackMut(); ok {
		t.Fatal("miss")
	}
	if _, _, ok := b.LookupAtMut(0); ok {
		t.Fatal("miss")
	}
	b.GetAtMut(0)
	if _, _, ok := b.PopFront(); ok {
		t.Fatal("miss")
	}
	if _, _, ok := b.PopBack(); ok {
		t.Fatal("miss")
	}
	for range b.AllMut() {
		t.Fatal("empty AllMut yielded")
	}
	for range b.BackwardMut() {
		t.Fatal("empty BackwardMut yielded")
	}
	for range b.AscendMut(0) {
		t.Fatal("empty AscendMut yielded")
	}
	for range b.DescendMut(0) {
		t.Fatal("empty DescendMut yielded")
	}
	for range b.AscendAtMut(0) {
		t.Fatal("empty AscendAtMut yielded")
	}
	for range b.ValuesMut() {
		t.Fatal("empty ValuesMut yielded")
	}
	if err := c.Err(); err != nil {
		t.Fatalf("cursor invalidated by empty mut ops: %v", err)
	}
	if ok, _ := c.First(); ok {
		t.Fatal("empty First")
	}
}

func TestMapCursorClearThenReuse(t *testing.T) {
	b, _ := newIntMap(100)
	c := b.Cursor()
	ok, _ := c.First()
	mustOK(t, ok, nil, "First")
	b.Clear()
	if err := c.Err(); err != ErrMapCursorInvalid {
		t.Fatalf("Clear must invalidate: %v", err)
	}
	assertCursorDead(t, c, "Clear")

	// A fresh cursor on the emptied map follows all empty-tree rules.
	c2 := b.Cursor()
	if ok, _ := c2.First(); ok || !c2.AfterLast() {
		t.Fatal("empty First after Clear")
	}
	if ok, _ := c2.Last(); ok || !c2.BeforeFirst() {
		t.Fatal("empty Last after Clear")
	}
	if _, _, ok := c2.Item(); ok {
		t.Fatal("empty Item after Clear")
	}
}

func TestMapCursorRelease(t *testing.T) {
	b, _ := newIntMap(50)
	c := b.Cursor()
	b.Release()
	if err := c.Err(); err != ErrMapCursorInvalid {
		t.Fatalf("Release must invalidate: %v", err)
	}
	assertCursorDead(t, c, "Release")
}

func TestMapCursorCloneIsolation(t *testing.T) {
	const n = 3000
	b, keys := newIntMap(n)

	// A cursor bound to the original stays valid across writes to a clone.
	c := b.Cursor()
	ok, err := c.SeekGE(1000)
	mustOK(t, ok, err, "SeekGE on original")
	clone := b.Clone()
	for i := 0; i < 200; i++ {
		clone.Set(5000+i, i)
		clone.Delete(i)
		clone.Replace(1000+i, -1)
	}
	if err := c.Err(); err != nil {
		t.Fatalf("writes on clone invalidated original cursor: %v", err)
	}
	curExpectInt(t, c, keys, 1000)
	got := make([]int, 0, n-1000)
	for {
		k, _, okk := c.Item()
		if !okk {
			break
		}
		got = append(got, k)
		cont, err := c.Next()
		if err != nil {
			t.Fatal(err)
		}
		if !cont {
			break
		}
	}
	if !slices.Equal(got, keys[1000:]) {
		t.Fatal("cursor scan diverged after clone writes")
	}

	// A write on the original invalidates its own cursor but not a cursor
	// bound to the clone.
	cc := clone.Cursor()
	b.Insert(9000, 1)
	if err := cc.Err(); err != nil {
		t.Fatalf("write on original invalidated clone cursor: %v", err)
	}
	if err := c.Err(); err != ErrMapCursorInvalid {
		t.Fatalf("write on original must invalidate its cursor: %v", err)
	}

	// A cursor on the clone is invalidated by clone writes only.
	cc2 := clone.Cursor()
	ok, _ = cc2.First()
	mustOK(t, ok, nil, "clone First")
	clone.Set(9001, 1)
	if err := cc2.Err(); err != ErrMapCursorInvalid {
		t.Fatalf("clone write must invalidate clone cursor: %v", err)
	}
}

func TestMapCursorNoCopyReleaseHooks(t *testing.T) {
	var copies, releases int
	b := NewMapOptions[int, int](MapOptions[int, int]{
		Copy:    func(v int) int { copies++; return v },
		Release: func(v int) { releases++ },
	})
	for i := 0; i < 500; i++ {
		b.Set(i, i*10)
	}
	_ = b.Clone() // arm copy-on-write so subsequent writes would exercise Copy
	copies, releases = 0, 0

	c := b.Cursor()
	if ok, _ := c.First(); !ok {
		t.Fatal("First")
	}
	if ok, _ := c.SeekGE(123); !ok {
		t.Fatal("SeekGE")
	}
	if ok, _ := c.Skip(50); !ok {
		t.Fatal("Skip")
	}
	if ok, _ := c.Skip(-50); !ok {
		t.Fatal("Skip back")
	}
	if ok, _ := c.Last(); !ok {
		t.Fatal("Last")
	}
	if ok, _ := c.Next(); ok {
		t.Fatal("Next at last should miss")
	}
	if ok, _ := c.Prev(); !ok {
		t.Fatal("Prev from after")
	}
	// A second coexisting cursor must be free of hooks as well.
	_ = b.Cursor()
	if copies != 0 || releases != 0 {
		t.Fatalf("read-only cursor invoked hooks: copies=%d releases=%d",
			copies, releases)
	}
}
