package exmat

import (
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

func TestString(t *testing.T) {
	m := NewExMat(2, 2, []float64{
		1, 2, 3, 4,
	})
	ans := "1.0000 2.0000 \n3.0000 4.0000 \n"
	if m.String() != ans {
		t.Error("not same")
		t.Log(m.String())
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
		t.Log(res)
	}
}

func TestPooling1(t *testing.T) {
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
		t.Log(res1.String())
	}

	t.Log("Test for average pooling.")
	var res2 ExMat
	res2.Pooling(2, 2, Avg, m)

	if !res2.Equals(ans2) {
		t.Error("Not same")
		t.Log(res2.String())
	}
}

func TestPooling2(t *testing.T) {
	m := NewExMat(5, 5, []float64{
		12, 20, 30, 0, 5,
		8, 12, 2, 0, 6,
		34, 70, 37, 4, 7,
		112, 100, 25, 12, 8,
		32, 64, 22, 100, 55,
	})
	ans := NewExMat(3, 3, []float64{
		12, 30, 5,
		34, 70, 7,
		112, 100, 100,
	})

	t.Log("Test for max pooling.")
	var res ExMat
	res.Pooling(2, 2, Max, m)

	if !res.Equals(ans) {
		t.Error("Not same")
		t.Log(res.String())
	}
}

func TestZeroPad(t *testing.T) {
	m1 := NewExMat(3, 4, []float64{
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1,
	})
	ans := NewExMat(5, 6, []float64{
		0, 0, 0, 0, 0, 0,
		0, 1, 1, 1, 1, 0,
		0, 1, 1, 1, 1, 0,
		0, 1, 1, 1, 1, 0,
		0, 0, 0, 0, 0, 0,
	})
	res := m1.ZeroPadding(1)
	if !res.Equals(ans) {
		t.Error("not same")
		t.Log(res)
	}

}

func TestEdgePad(t *testing.T) {
	m1 := NewExMat(3, 4, []float64{
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1,
	})
	ans := NewExMat(5, 6, []float64{
		1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1,
	})
	res := m1.EdgePadding(1)
	if !res.Equals(ans) {
		t.Error("not same")
		t.Log(res)
	}
}

func TestFlatten(t *testing.T) {
	m1 := NewExMat(3, 4, []float64{
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1,
	})

	ans1 := NewExMat(1, 12, []float64{
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	})

	f1 := m1.Flatten()
	if !ans1.Equals(f1) {
		t.Error("Not same")
		t.Log(f1)
	}

	m2 := NewExMat(3, 6, []float64{
		1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1,
		2, 2, 2, 2, 2, 2,
	})

	ans2 := NewExMat(1, 18, []float64{
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
	})

	f2 := m2.Flatten()

	if !ans2.Equals(f2) {
		t.Error("Not same")
		t.Log(f2)
	}
}

func TestIm2Col(t *testing.T) {
	m := NewExMat(7, 7, []float64{
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 2, 0, 0, 1, 0,
		0, 1, 2, 0, 0, 1, 0,
		0, 2, 2, 1, 2, 2, 0,
		0, 0, 0, 1, 2, 1, 0,
		0, 2, 1, 1, 1, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
	})

	a := NewExMat(9, 9, []float64{
		0, 0, 0, 0, 0, 2, 0, 1, 2,
		0, 0, 0, 2, 0, 0, 2, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 1, 0,
		0, 1, 2, 0, 2, 2, 0, 0, 0,
		2, 0, 0, 2, 1, 2, 0, 1, 2,
		0, 1, 0, 2, 2, 0, 2, 1, 0,
		0, 0, 0, 0, 2, 1, 0, 0, 0,
		0, 1, 2, 1, 1, 1, 0, 0, 0,
		2, 1, 0, 1, 0, 0, 0, 0, 0,
	})
	res := m.Im2Col(3, 2)

	if !res.Equals(a) {
		t.Error("not same")
		t.Log(res)
	}
}

func TestConvolve2d(t *testing.T) {
	m := NewExMat(5, 5, []float64{
		1, 1, 1, 0, 0,
		0, 1, 1, 1, 0,
		0, 0, 1, 1, 1,
		0, 0, 1, 1, 0,
		0, 1, 1, 0, 0,
	})
	f := NewExMat(3, 3, []float64{
		1, 0, 1,
		0, 1, 0,
		1, 0, 1,
	})
	ans := NewExMat(3, 3, []float64{
		4, 3, 4,
		2, 4, 3,
		2, 3, 4,
	})
	err := m.Convolve2d(1, f)
	if err != nil {
		t.Fatal(err)
	}

	if !m.Equals(ans) {
		t.Error("not same")
		t.Log(m)
	}
}
