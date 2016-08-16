package exmat

import (
	"bytes"
	"strconv"
	"testing"
)

func TestInitializer(t *testing.T) {
	m := NewExMat(2, 2, []float64{1, 2, 3, 4})
	if m.At(0, 0) != 1 {
		t.Error("not same")
	}
	if m.At(0, 1) != 2 {
		t.Error("not same")
	}
	if m.At(1, 0) != 3 {
		t.Error("not same")
	}
	if m.At(1, 1) != 4 {
		t.Error("not same")
	}
}

func TestEquals(t *testing.T) {
	m1 := NewExMat(2, 2, []float64{
		1, 1, 1, 1,
	})
	m2 := NewExMat(2, 2, []float64{
		2, 2, 2, 2,
	})
	if !m1.Equals(m1) {
		t.Error("not same")
	}
	if m1.Equals(m2) {
		t.Error("same")
	}
}

func TestReshape(t *testing.T) {
	m := NewExMat(4, 3, []float64{
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
		0, 0, 0,
	})

	ans := NewExMat(2, 6, []float64{
		0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
	})

	var res ExMat
	err := res.Reshape(2, 6, m)
	if err != nil {
		t.Error(err)
	}

	if !res.Equals(ans) {
		t.Errorf("Not same.")
		t.Log(showExMat(&res))
	}
}

func showExMat(emat *ExMat) string {
	var w bytes.Buffer
	r, c := emat.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			w.WriteString(strconv.FormatFloat(emat.At(i, j), 'f', 4, 64))
			w.WriteString(", ")
		}
		w.WriteString("\n")
	}
	return w.String()
}

func TestPooling(t *testing.T) {
	m := NewExMat(4, 4, []float64{
		12, 20, 30, 0,
		8, 12, 2, 0,
		34, 70, 37, 4,
		112, 100, 25, 12,
	})
	ans1 := NewExMat(2, 2, []float64{
		20, 30,
		112, 37,
	})

	ans2 := NewExMat(2, 2, []float64{
		13, 8,
		79, 19.5,
	})

	t.Log("Test for max pooling.")
	var res1 ExMat
	res1.Pooling(2, 2, Max, m)

	if !res1.Equals(ans1) {
		t.Error("Not same")
		t.Log(showExMat(&res1))
	}

	t.Log("Test for average pooling.")
	var res2 ExMat
	res2.Pooling(2, 2, Avg, m)

	if !res2.Equals(ans2) {
		t.Error("Not same")
		t.Log(showExMat(&res2))
	}
}

func TestEdgePad(t *testing.T) {
	m1 := NewExMat(3, 4, []float64{
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1,
	})
	ans := NewExMat(4, 5, []float64{
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
	})
	var res ExMat
	res.EdgePadding(1, m1)
	if !res.Equals(ans) {
		t.Error("not same")
	}
}
