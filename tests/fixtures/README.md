# rdrobust Senate test fixture

Source: [official rdrobust example data](https://github.com/rdpackages/rdrobust/blob/main/stata/rdrobust_senate.dta).
This CSV retains only the `margin` and `vote` columns of the previously downloaded local `rdrobust_senate.dta`, without transforming their values. It is an election example, not CHARLS individual data. It makes the published-example test independent of a file outside this replication directory.
