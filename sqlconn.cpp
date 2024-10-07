#include "sqlconn.h"

int SQLConn::openDB(const std::string &dbFile)
{
	sqlite3 *sql;
	char *err;
	int ret;

	ret = sqlite3_open_v2(dbFile.c_str(), &sql, SQLITE_OPEN_READWRITE |
			      SQLITE_OPEN_CREATE, NULL);
	sqlHolder.reset(sql);
	if (ret != SQLITE_OK) {
		std::cerr << "db open failed: " << sqlite3_errstr(ret) << "\n";
		return -1;
	}

	ret = sqlite3_exec(sqlHolder, "PRAGMA foreign_keys = ON;", NULL, NULL,
			   &err);
	if (ret != SQLITE_OK) {
		std::cerr << "db PRAGMA failed (" << __LINE__ << "): " <<
				sqlite3_errstr(ret) << " -> " << err << "\n";
		sqlite3_free(err);
		return -1;
	}

	return 0;
}

int SQLConn::prepDB()
{
	sqlite3_stmt *stmt;
	int ret;

	ret = sqlite3_prepare_v2(sqlHolder,
				 "SELECT map.user, map.file, sum(map.count) AS count, user.email, "
				 "dir.dir || '/' || file.file AS path "
				 "FROM user_file_map AS map "
				 "LEFT JOIN user ON map.user=user.id "
				 "LEFT JOIN file ON map.file=file.id "
				 "LEFT JOIN dir ON file.dir=dir.id "
				 "GROUP BY map.user, map.file "
				 "ORDER BY random();",
				 -1, &stmt, NULL);
	selMap.reset(stmt);
	if (ret != SQLITE_OK) {
		std::cerr << "db prepare failed (" << __LINE__ << "): " <<
			     sqlite3_errstr(ret) << " -> " <<
			     sqlite3_errmsg(sqlHolder) << "\n";
		return -1;
	}

	return 0;
}

int SQLConn::open(const std::string &dbFile)
{
	if (openDB(dbFile) < 0)
		return -1;
	if (prepDB() < 0)
		return -1;

	return 0;
}
