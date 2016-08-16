package exmat

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"math"
	"math/rand"
	"sync"
	"time"
)

// PadMode is flag for padding mode.
type PadMode int

const (
	// Zero is zero padding flag.
	Zero = iota + 1
	// Edge is edge padding flag.
	Edge
)

// PoolingMode is flag for pooling.
type PoolingMode int

const (
	// Max pooling
	Max = iota + 1
	// Avg shows average pooling.
	Avg
)

// ExMat is extented Matrix of gonum/matrix
type ExMat struct {
	*mat64.Dense
}

// NewExMat is constructor
func NewExMat(r, c int, src []float64) *ExMat {
	return &ExMat{
		mat64.NewDense(r, c, src),
	}
}

// Random generates random ExMat.
func Random(r, c int) *ExMat {
	rand.Seed(time.Now().UnixNano())
	res := make([]float64, r*c)
	for y := 0; y < int(r); y++ {
		for x := 0; x < int(c); x++ {
			res[y*c+x] = rand.Float64()
		}
	}
	return NewExMat(r, c, res)
}

// Zeros generates zero matrix.
func Zeros(r, c int) *ExMat {
	rand.Seed(time.Now().UnixNano())
	res := make([]float64, r*c)
	for y := 0; y < r; y++ {
		for x := 0; x < c; x++ {
			res[y*c+x] = 0.0
		}
	}
	return NewExMat(r, c, res)
}

func cmp(a, b []float64, resCh chan bool) {
	for i := range a {
		if a[i] != b[i] {
			resCh <- false
		}
	}
	resCh <- true
}

func (emat *ExMat) Equals(t *ExMat) bool {
	rs, cs := emat.Dims()
	rt, ct := t.Dims()
	if rs != rt || cs != ct {
		return false
	}
	ch := make(chan bool, rs)
	for y := 0; y < rs; y++ {
		go cmp(emat.RawRowView(y), t.RawRowView(y), ch)
	}
	for y := 0; y < rs; y++ {
		if res := <-ch; !res {
			return false
		}
	}
	close(ch)
	return true
}

// Reshape reshapes matrix
func (emat *ExMat) Reshape(r, c int, src *ExMat) (err error) {
	rs, cs := src.Dims()
	if rs*cs != r*c {
		return fmt.Errorf("Size is not same")
	}
	row := 0
	col := 0
	newMat := make([]float64, r*c)
	for y := 0; y < r; y++ {
		for x := 0; x < c; x++ {
			if col > cs-1 {
				row++
				col = 0
			}
			newMat[y*c+x] = src.At(row, col)
			col++
		}
	}
	emat.Dense = mat64.NewDense(r, c, newMat)
	return
}

func maxPool(m mat64.Matrix) float64 {
	r, c := m.Dims()
	max := m.At(0, 0)
	for y := 0; y < r; y++ {
		for x := 0; x < c; x++ {
			if max < m.At(y, x) {
				max = m.At(y, x)
			}
		}
	}
	return max
}

func avgPool(m mat64.Matrix) float64 {
	r, c := m.Dims()
	sum := 0.0
	for y := 0; y < r; y++ {
		for x := 0; x < c; x++ {
			sum += m.At(y, x)
		}
	}
	return sum / float64(r*c)
}

func (emat *ExMat) execPool(newMat []float64, y, rows, cols, k, s int, mode PoolingMode, wg *sync.WaitGroup) {
	for x := 0; x < cols; x++ {
		part := emat.View(y*s, x*s, k, k)
		switch mode {
		case Max:
			newMat[y*cols+x] = maxPool(part)
		case Avg:
			newMat[y*cols+x] = avgPool(part)
		}
	}
	wg.Done()
}

// Pooling pool the matrix
func (emat *ExMat) Pooling(k, s int, mode PoolingMode, src *ExMat) {
	rs, cs := src.Dims()
	rows := int(math.Ceil(float64(rs-k)/float64(s))) + 1
	cols := int(math.Ceil(float64(cs-k)/float64(s))) + 1
	if (rs-k)%s != 0 {
		padded := src.Grow(1, 1)
		src.Dense = padded.(*mat64.Dense)
	}
	newMat := make([]float64, rows*cols)
	var wg sync.WaitGroup
	for y := 0; y < rows; y++ {
		wg.Add(1)
		go src.execPool(newMat, y, rows, cols, k, s, mode, &wg)
	}
	wg.Wait()
	emat.Dense = mat64.NewDense(rows, cols, newMat)
}

func (emat *ExMat) flatten(res []float64, y int, wg *sync.WaitGroup) {
	r, c := emat.Dims()
	for x := 0; x < c; x++ {
		res[y*c+x] = emat.At(y, x)
	}
	wg.Done()
}

func (emat *ExMat) Flatten() mat64.Matrix {
	r, c := emat.Dims()
	res := make([]float64, r*c)
	var wg sync.WaitGroup
	for y := 0; y < r; y++ {
		wg.Add(1)
		go emat.flatten(res, y, &wg)
	}
	wg.Wait()
	return mat64.NewDense(1, r*c, res)
}

func (emat *ExMat) edgePad(rows, cols, newRows, newCols int, newMatrix []float64, w, y int, wg *sync.WaitGroup) {
	for x := 0; x < newCols; x++ {
		if y < w && x < w {
			newMatrix[y*newCols+x] = emat.At(0, 0)
		} else if y < w && x > w-1 && x < cols+w {
			newMatrix[y*newCols+x] = emat.At(0, x-w)
		} else if y < w && x > cols+w-1 {
			newMatrix[y*newCols+x] = emat.At(0, cols-1)
		} else if y > w-1 && y < rows+w && x < w {
			newMatrix[y*newCols+x] = emat.At(y-w, 0)
		} else if y > w-1 && y < rows+w && x > cols+w-1 {
			newMatrix[y*newCols+x] = emat.At(y-w, cols-1)
		} else if y > rows+w-1 && x < w {
			newMatrix[y*newCols+x] = emat.At(rows-1, 0)
		} else if y > rows+w-1 && x > w-1 && x < cols+w {
			newMatrix[y*newCols+x] = emat.At(rows-1, x-w)
		} else if y > rows+w-1 && x > cols+w-1 {
			newMatrix[y*newCols+x] = emat.At(rows-1, cols-1)
		} else {
			newMatrix[y*newCols+x] = emat.At(y-w, x-w)
		}
	}
	wg.Done()
}

func (emat *ExMat) EdgePadding(w int, src *ExMat) {
	r, c := src.Dims()
	newRows := r + w*2
	newCols := c + w*2
	newMatrix := make([]float64, newRows*newCols)
	var wg sync.WaitGroup
	for y := 0; y < newRows; y++ {
		wg.Add(1)
		go emat.edgePad(r, c, newRows, newCols, newMatrix, w, y, &wg)
	}
	wg.Wait()
	emat.Dense = mat64.NewDense(newRows, newCols, newMatrix)
}

func (emat *ExMat) Im2Col(k, s int) Matrix {
	colSize := k * k
	r, c := emat.Dims()
	rows := (r-k)/s + 1
	cols := (c-k)/s + 1
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			sy := y * s
			sx := x * s
			part := emat.View(sy, sx, k, k).(*mat64.Dense)
			var partExMat ExMat
			partExMat.Dense = part
			flat := partExMat.Flatten()
		}
	}
}

// Matrix is object for matrix.
type Matrix struct {
	Rows uint
	Cols uint
	M    [][]float32
}

// Dot calculate dot of two vector.
func Dot(v1, v2 []float32) (float32, error) {
	if len(v1) != len(v2) {
		return 0.0, fmt.Errorf("Length mismatched %d, %d\n", len(v1), len(v2))
	}
	sum := float32(0.0)
	for i := 0; i < len(v1); i++ {
		sum += v1[i] * v2[i]
	}
	return sum, nil
}

// Dot2d calculate dot of a matrix.
func Dot2d(m1, m2 [][]float32) (float32, error) {
	sum := float32(0.0)
	for i := 0; i < len(m1); i++ {
		partial, err := Dot(m1[i], m2[i])
		if err != nil {
			return 0.0, err
		}
		sum += partial
	}
	return sum, nil
}

// Slice2d slices a matrix.
func Slice2d(s [][]float32, rs, re, cs, ce uint) [][]float32 {
	sr := make([][]float32, re-rs)
	copy(sr, s[rs:re])
	for y := 0; y < len(sr); y++ {
		sr[y] = sr[y][cs:ce]
	}
	return sr
}

func (m *Matrix) execConv(newMat [][]float32, f *Matrix, y int, cols, rows, stride uint, errCh chan error) {
	newMat[y] = make([]float32, cols)
	var err error
	for x := 0; x < int(cols); x++ {
		newMat[y][x], err = Dot2d(Slice2d(m.M, uint(y)*stride, uint(y)*stride+f.Rows, uint(x)*stride, uint(x)*stride+f.Cols), f.M)
		if err != nil {
			errCh <- err
		}
	}
	errCh <- nil
}

// Convolve2d convolve a 2d matrix.
func (m *Matrix) Convolve2d(f *Matrix, stride, pad uint, mode PadMode) (*Matrix, error) {
	rows := (m.Rows-f.Rows+2*pad)/stride + 1
	cols := (m.Cols-f.Rows+2*pad)/stride + 1
	if pad > 0 {
		m = m.Pad(pad, mode)
	}
	newMat := make([][]float32, rows)
	errCh := make(chan error, cols)
	for y := 0; y < int(rows); y++ {
		go m.execConv(newMat, f, y, cols, rows, stride, errCh)
	}

	for i := 0; i < int(rows); i++ {
		err := <-errCh
		if err != nil {
			return nil, err
		}
	}
	close(errCh)
	return NewMatrix(newMat), nil
}

func (m *Matrix) t(newMat [][]float32, y int, wg *sync.WaitGroup) {
	col := make([]float32, m.Rows)
	for x := 0; x < int(m.Rows); x++ {
		col[x] = m.M[x][y]
	}
	newMat[y] = col
	wg.Done()
}

func makeCol(m [][]float32, colSize, rs, cs, kernelSize uint) []float32 {
	col := make([]float32, colSize)
	idx := 0
	for y := rs; y < rs+kernelSize; y++ {
		for x := cs; x < cs+kernelSize; x++ {
			col[idx] = m[y][x]
			idx++
		}
	}
	return col
}

// Im2Col make clumns matrix from matrix.
func (m *Matrix) Im2Col(kernelSize, stride uint) *Matrix {
	colSize := kernelSize * kernelSize
	var res [][]float32
	for y := 0; y < int(m.Rows-kernelSize+1); y += int(stride) {
		for x := 0; x < int(m.Cols-kernelSize+1); x += int(stride) {
			res = append(res, makeCol(m.M, colSize, uint(y), uint(x), kernelSize))
		}
	}
	return NewMatrix(res)
}
