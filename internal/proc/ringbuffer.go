package proc

import "sync"

// ringBuffer is a bounded, append-only byte buffer that discards from the
// front once it exceeds capacity, keeping the most recent output — the
// right trade-off for a live process's log: an old-and-full buffer should
// lose its oldest lines, not refuse new ones or grow forever.
//
// Discarding from the front (not overwriting in place) is what makes
// Since's absolute-offset cursor safe: a cursor pointing at already-dropped
// data is simply clamped forward, rather than reading corrupted bytes a
// true circular buffer would leave behind.
type ringBuffer struct {
	mu       sync.Mutex
	data     []byte
	dropped  int64 // total bytes ever discarded from the front
	capacity int
}

func newRingBuffer(capacity int) *ringBuffer {
	return &ringBuffer{capacity: capacity}
}

// Write implements io.Writer, so a ringBuffer can be used directly as
// exec.Cmd's Stdout/Stderr.
func (r *ringBuffer) Write(p []byte) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.data = append(r.data, p...)
	if excess := len(r.data) - r.capacity; excess > 0 {
		r.data = r.data[excess:]
		r.dropped += int64(excess)
	}
	return len(p), nil
}

// Since returns everything written after the absolute offset from, and the
// offset to pass next time to continue reading only what's new.
func (r *ringBuffer) Since(from int64) (data []byte, next int64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if from < r.dropped {
		from = r.dropped // some of what was requested was already discarded
	}
	start := from - r.dropped
	out := make([]byte, len(r.data)-int(start))
	copy(out, r.data[start:])
	return out, r.dropped + int64(len(r.data))
}
