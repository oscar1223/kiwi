package proc

import "testing"

func TestRingBufferReadsWhatWasWritten(t *testing.T) {
	r := newRingBuffer(1024)
	r.Write([]byte("hello "))
	r.Write([]byte("world"))

	got, next := r.Since(0)
	if string(got) != "hello world" {
		t.Errorf("got %q", got)
	}
	if next != 11 {
		t.Errorf("next = %d, want 11", next)
	}
}

func TestRingBufferSinceOnlyReturnsNewData(t *testing.T) {
	r := newRingBuffer(1024)
	r.Write([]byte("first"))
	_, next := r.Since(0)

	r.Write([]byte("second"))
	got, next2 := r.Since(next)
	if string(got) != "second" {
		t.Errorf("got %q, want %q", got, "second")
	}
	if next2 != 11 {
		t.Errorf("next2 = %d, want 11", next2)
	}
}

func TestRingBufferSinceAtCurrentOffsetIsEmpty(t *testing.T) {
	r := newRingBuffer(1024)
	r.Write([]byte("data"))
	_, next := r.Since(0)

	got, _ := r.Since(next)
	if len(got) != 0 {
		t.Errorf("got %q, want empty", got)
	}
}

// The core property: once the buffer is over capacity, it drops the oldest
// bytes and a stale cursor is clamped forward rather than reading garbage or
// panicking on a negative slice index.
func TestRingBufferDropsOldestBytesOverCapacity(t *testing.T) {
	r := newRingBuffer(10)
	r.Write([]byte("0123456789")) // exactly at capacity
	r.Write([]byte("ABCDE"))      // pushes the first 5 bytes out

	got, next := r.Since(0)
	if string(got) != "56789ABCDE" {
		t.Errorf("got %q, want the most recent 10 bytes", got)
	}
	if next != 15 {
		t.Errorf("next = %d, want 15", next)
	}
}

func TestRingBufferClampsAStaleCursorForward(t *testing.T) {
	r := newRingBuffer(5)
	r.Write([]byte("aaaaa"))
	_, cursorBeforeDrop := r.Since(0) // cursor = 5, nothing dropped yet

	r.Write([]byte("bbbbb")) // this drops all of "aaaaa"

	// Reading from the pre-drop cursor must not panic or return dropped
	// bytes — it should clamp to what's actually still available.
	got, _ := r.Since(cursorBeforeDrop)
	if string(got) != "bbbbb" {
		t.Errorf("got %q, want %q", got, "bbbbb")
	}
}

func TestRingBufferEmptyRead(t *testing.T) {
	r := newRingBuffer(1024)
	got, next := r.Since(0)
	if len(got) != 0 || next != 0 {
		t.Errorf("got (%q, %d), want (\"\", 0)", got, next)
	}
}

// Concurrent writers (stdout+stderr piped into the same buffer) and a
// concurrent reader must not race — this is exactly how exec.Cmd uses it.
func TestRingBufferConcurrentWritesDoNotRace(t *testing.T) {
	r := newRingBuffer(1024)
	done := make(chan struct{})
	go func() {
		for i := 0; i < 100; i++ {
			r.Write([]byte("x"))
		}
		close(done)
	}()
	for i := 0; i < 100; i++ {
		r.Write([]byte("y"))
	}
	<-done

	got, _ := r.Since(0)
	if len(got) != 200 {
		t.Errorf("got %d bytes, want 200", len(got))
	}
}
