#ifndef SQLCONN_H
#define SQLCONN_H

#include "sqlite.h"

class SQLConn {
public:
	SQLConn() {}

	int open(const std::string &dbFile);

	int begin();
	int end();

	sqlite3 *getSql() { return sqlHolder.get(); }
	sqlite3_stmt *getSelMap() { return selMap.get(); }
private:
	int openDB(const std::string &dbFile);
	int prepDB();

	SQLHolder sqlHolder;
	SQLStmtHolder selMap;
};

#endif // SQLCONN_H
