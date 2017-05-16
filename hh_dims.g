LoadPackage("qpa");

# Based on https://github.com/gap-packages/qpa/blob/115dcc995412b52d0b69f7212fb0293e62ed7999/lib/modulehomalg.gi#L727
ExtAlgebraDimensions := function(M, n)
    local N, projcovers, f, i, EndM, J, gens, extgroups, dim_ext_groups;

    K := LeftActingDomain(M);
    N := M;
    projcovers := [];
    for i in [1..n] do
        f := ProjectiveCover(N);
        Add(projcovers,f);
        N := Kernel(f);
    od;
    extgroups := [];
    #
    #   Computing Ext^i(M,M) for i = 0, 1, 2,...., n.
    #
    for i in [0..n] do
        if i = 0 then
            EndM := EndOverAlgebra(M);
            J := RadicalOfAlgebra(EndM);
            gens := GeneratorsOfAlgebra(J);
            Add(extgroups, [[],List(gens, x -> FromEndMToHomMM(M,x))]);
        elif i = 1 then
            Add(extgroups, ExtOverAlgebra(M,M));
        else
            Add(extgroups, ExtOverAlgebra(Kernel(projcovers[i-1]),M));
        fi;
    od;
    dim_ext_groups := List(extgroups, x -> Length(x[2]));
    dim_ext_groups[1] := Dimension(EndM);
#    Print(Dimension(EndM) - Dimension(J)," generators in degree 0.\n");
    return [dim_ext_groups];
end;

Q := Quiver(1, [ [1,1,"x"], [1,1,"y"] ]);
kQ := PathAlgebra(GF(2), Q);
AssignGeneratorVariables(kQ);

MyAlgebraDimensions := function(kQ, k, c, d, n)
    local relations, A, M;

    relations := [
        x^2 - y*(x*y)^(k-1) - c*(x*y)^k,
        y^2 - d*(x*y)^k,
        (x*y)^k - (y*x)^k,
        x*(y*x)^k
    ];

    A := kQ / relations;

    M := AlgebraAsModuleOverEnvelopingAlgebra(A);
    # projres := ProjectiveResolution(M);
    # ObjectOfComplex(projres, 5);
    # ExtAlgebraGenerators(M, 3);

    return ExtAlgebraDimensions(M, n);
end;

k := 2;
c := One(GF(2));
d := One(GF(2));

for k in [2..7] do
    for c in [Zero(GF(2)), One(GF(2))] do
        for d in [Zero(GF(2)), One(GF(2))] do
            Print("k = ", k);
            if c = Zero(GF(2)) then
                Print(", c = 0");
            else
                Print(", c = 1");
            fi;
            if d = Zero(GF(2)) then
                Print(", d = 0");
            else
                Print(", d = 1");
            fi;
            Print("    ");
            start := Runtime();
            Print(MyAlgebraDimensions(kQ, k, c, d, 7));
            runtime := Runtime() - start;
            Print("    ", runtime, "\n");
        od;
    od;
od;

# k = 2, c = 0, d = 0    [ [ 5, 8, 9, 10, 13, 16, 17, 18 ] ]    41770
# k = 2, c = 0, d = 1    [ [ 5, 7, 7, 6, 7, 9, 9, 8 ] ]    40417
# k = 2, c = 1, d = 0    [ [ 5, 8, 9, 10, 13, 16, 17, 18 ] ]    41910
# k = 2, c = 1, d = 1    [ [ 5, 7, 7, 6, 7, 9, 9, 8 ] ]    41127
# k = 3, c = 0, d = 0    [ [ 6, 9, 10, 11, 14, 17, 18, 19 ] ]    691396
# k = 3, c = 0, d = 1    [ [ 6, 7, 7, 7, 8, 9, 9, 9 ] ]    673937
# k = 3, c = 1, d = 0    [ [ 6, 8, 9, 11, 14, 16, 17, 19 ] ]    701337
# k = 3, c = 1, d = 1    [ [ 6, 7, 7, 7, 8, 9, 9, 9 ] ]    705290
# k = 4, c = 0, d = 0    [ [ 7, 10, 11, 12, 15, 18, 19, 20 ] ]    7354412
# k = 4, c = 0, d = 1    [ [ 7, 9, 9, 8, 9, 11, 11, 10 ] ]    7374119
# k = 4, c = 1, d = 0    [ [ 7, 10, 11, 12, 15, 18, 19, 20 ] ]    7406926
# k = 4, c = 1, d = 1    [ [ 7, 9, 9, 8, 9, 11, 11, 10 ] ]    7641130
