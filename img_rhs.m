function rhs = img_rhs(t,A,dummy,L,D)
    rhs = D.*(L*A);